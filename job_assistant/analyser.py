"""
analyser.py
Gap analysis between a job spec and a CV.
Uses two smaller LLM calls for job spec parsing to avoid context limit issues.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm import chat


# ── Schemas — kept small to stay within model context limits ──────────────────

JOB_SCHEMA_BASIC = {
    "job_title": "string",
    "organisation": "string",
    "word_limit": "integer or null",
    "role_summary": "one sentence summary of the role"
}

JOB_SCHEMA_REQUIREMENTS = {
    "sift_criteria": ["string"],
    "required_experience": ["string"],
    "technical_skills": ["string"],
    "behaviours": ["string"]
}

CV_SCHEMA = {
    "name": "string",
    "current_role": "string",
    "current_employer": "string",
    "technical_skills": ["string"],
    "domains": ["string"],
    "qualifications": ["string"],
    "key_achievements": ["string"]
}

# Note: no overall_fit_score — local models anchor to 80 regardless of evidence.
# Score is computed from match counts in analyse_gaps() instead.
GAP_SCHEMA = {
    "strong_matches": [
        {"requirement": "string", "evidence": "string"}
    ],
    "partial_matches": [
        {"requirement": "string", "gap": "string"}
    ],
    "missing": [
        {"requirement": "string", "why_it_matters": "string"}
    ],
    "fit_summary": "string",
    "priority_gaps": ["string"]
}


# ── JSON extraction ───────────────────────────────────────────────────────────

def _extract_json(prompt: str, schema: dict, retries: int = 2) -> dict:
    """Ask the LLM to return JSON matching a schema. Retries on parse failure."""
    system = (
        "You return ONLY valid JSON. "
        "No explanation, no markdown fences, no extra text. "
        "Return a single JSON object matching the schema exactly."
    )
    full_prompt = (
        f"{prompt}\n\n"
        f"Return ONLY a JSON object with these exact keys:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        f"No other text. No markdown. Just the JSON."
    )
    for attempt in range(retries + 1):
        raw = chat(full_prompt, system=system, temperature=0.0)
        try:
            text = raw.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1]).strip()
            if not text.startswith("{"):
                start = text.find("{")
                end = text.rfind("}") + 1
                if start != -1 and end > start:
                    text = text[start:end]
            return json.loads(text)
        except json.JSONDecodeError as e:
            if attempt < retries:
                print(f"  Parse attempt {attempt + 1} failed ({e}), retrying...")
            else:
                print(f"  Could not parse JSON after {retries + 1} attempts: {e}")
                print(f"  Raw output: {raw[:400]}")
                return {}


def _compute_score(gap_result: dict) -> int:
    """
    Derive a fit score from match counts rather than asking the model.
    Avoids the model anchoring to 80 regardless of evidence.
    """
    strong = len(gap_result.get("strong_matches", []))
    partial = len(gap_result.get("partial_matches", []))
    missing = len(gap_result.get("missing", []))
    total = strong + partial + missing
    if total == 0:
        return 0
    score = int(((strong + 0.5 * partial) / total) * 100)
    return min(score, 99)  # cap at 99 — nothing is perfect


# ── Public API ────────────────────────────────────────────────────────────────

def parse_job_spec_structured(job_text: str) -> dict:
    """
    Extract structured data from a job spec.
    Two smaller calls to avoid context limit issues with local models.
    """
    print("Parsing job spec...")

    basic = _extract_json(
        f"Extract the job title, organisation name, word limit for personal statement "
        f"(if stated, otherwise null), and a one sentence role summary from this:\n\n{job_text}",
        JOB_SCHEMA_BASIC
    )

    reqs = _extract_json(
        f"From this job specification, extract four lists:\n"
        f"1. sift_criteria: what is assessed at sift stage\n"
        f"2. required_experience: required experience items\n"
        f"3. technical_skills: specific tools or technologies named\n"
        f"4. behaviours: behaviours or competencies assessed\n\n{job_text}",
        JOB_SCHEMA_REQUIREMENTS
    )

    result = {**basic, **reqs}
    print(f"  {result.get('job_title', '?')} at {result.get('organisation', '?')}")
    return result


def parse_cv_structured(cv_text: str) -> dict:
    """Extract structured data from a CV."""
    print("Parsing CV...")
    result = _extract_json(
        f"From this CV extract: name, current role, current employer, "
        f"list of technical skills, list of domain areas, list of qualifications, "
        f"and list of key achievements:\n\n{cv_text}",
        CV_SCHEMA
    )
    print(f"  {result.get('name', '?')} — {result.get('current_role', '?')}")
    return result


def analyse_gaps(parsed_job: dict, parsed_cv: dict, cv_text: str = "") -> dict:
    """
    Produce a gap analysis between a parsed job spec and a CV.
    Passes raw CV text to the model rather than a compressed summary,
    so it has enough context to identify genuine gaps vs false negatives.
    """
    print("Running gap analysis...")

    job_summary = (
        f"Role: {parsed_job.get('job_title', '')} at {parsed_job.get('organisation', '')}\n"
        f"Sift criteria: {parsed_job.get('sift_criteria', [])}\n"
        f"Required experience: {parsed_job.get('required_experience', [])}\n"
        f"Technical skills needed: {parsed_job.get('technical_skills', [])}"
    )

    # Use raw CV text if available for richer context, else fall back to parsed summary
    if cv_text.strip():
        cv_content = f"FULL CV:\n{cv_text}"
    else:
        cv_content = (
            f"Candidate: {parsed_cv.get('name', '')} — "
            f"{parsed_cv.get('current_role', '')} at {parsed_cv.get('current_employer', '')}\n"
            f"Technical skills: {parsed_cv.get('technical_skills', [])}\n"
            f"Domains: {parsed_cv.get('domains', [])}\n"
            f"Key achievements: {parsed_cv.get('key_achievements', [])}"
        )

    prompt = (
        f"You are a recruitment consultant doing a thorough gap analysis.\n"
        f"Read the CV carefully before identifying gaps — do not flag something as "
        f"missing if it is clearly evidenced in the CV text.\n\n"
        f"JOB REQUIREMENTS:\n{job_summary}\n\n"
        f"{cv_content}\n\n"
        f"Identify: strong matches with specific evidence from the CV, "
        f"partial matches where evidence exists but is thin, "
        f"genuine gaps where there is no evidence, "
        f"a brief honest summary, and the top 3 priority gaps to address."
    )

    result = _extract_json(prompt, GAP_SCHEMA)

    # Compute score from match counts rather than asking the model
    score = _compute_score(result)
    result["overall_fit_score"] = score

    strong = len(result.get("strong_matches", []))
    partial = len(result.get("partial_matches", []))
    missing = len(result.get("missing", []))
    print(f"  Fit score: {score}/100 ({strong} strong, {partial} partial, {missing} missing)")
    return result