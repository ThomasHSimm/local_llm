"""
writer.py
Draft and refine a personal statement using gap analysis
and elicited evidence. Enforces word count limits.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm import chat


def count_words(text: str) -> int:
    """Count words in a string."""
    return len(text.split())


def draft_statement(
    parsed_job: dict,
    parsed_cv: dict,
    gap_analysis: dict,
    evidence: list,
    word_limit: int = 500,
) -> str:
    """
    Draft a personal statement using job spec, CV, gap analysis and evidence.
    Targets the word limit and trims if over.
    """
    sift_criteria = parsed_job.get("sift_criteria", [])
    strong_matches = [
        f"{m['requirement']}: {m['evidence']}"
        for m in gap_analysis.get("strong_matches", [])
    ]

    evidence_text = "\n\n".join([
        f"Re: {e['gap']}\nQ: {e['question']}\nA: {e['answer']}"
        for e in evidence
    ])

    prompt = (
        "You are an expert job application writer. Draft a personal statement "
        "using ONLY the evidence provided below. Do not invent details.\n\n"
        f"ROLE: {parsed_job.get('job_title', '')} at {parsed_job.get('organisation', '')}\n\n"
        f"WHAT THE SIFT ASSESSES:\n{json.dumps(sift_criteria, indent=2)}\n\n"
        f"CANDIDATE STRENGTHS:\n{json.dumps(strong_matches, indent=2)}\n\n"
        f"ELICITED EVIDENCE:\n{evidence_text}\n\n"
        f"CURRENT ROLE: {parsed_cv.get('current_role', '')} "
        f"at {parsed_cv.get('current_employer', '')}\n\n"
        f"REQUIREMENTS:\n"
        f"- Maximum {word_limit} words\n"
        f"- Structure around the sift criteria\n"
        f"- Use specific examples — no vague claims\n"
        f"- First person, confident and direct tone\n"
        f"- Flowing prose — no bullet points\n"
        f"- Do not overclaim thin evidence\n\n"
        f"Return the personal statement as plain text only."
    )

    print(f"Drafting personal statement (target: {word_limit} words)...")
    draft = chat(prompt, temperature=0.4)
    wc = count_words(draft)
    print(f"  Draft: {wc} words (limit: {word_limit})")

    if wc > word_limit:
        print(f"  Over by {wc - word_limit} words — trimming...")
        draft = trim_to_limit(draft, word_limit)

    return draft


def trim_to_limit(draft: str, word_limit: int) -> str:
    """Ask the LLM to trim a draft to within the word limit."""
    over_by = count_words(draft) - word_limit
    prompt = (
        f"This personal statement is {over_by} words over the {word_limit} word limit.\n"
        f"Trim it to under {word_limit} words. Preserve all key examples and evidence. "
        f"Do not add any new content. Return the full trimmed statement only.\n\n"
        f"{draft}"
    )
    trimmed = chat(prompt, temperature=0.2)
    wc = count_words(trimmed)
    print(f"  Trimmed to {wc} words")
    return trimmed


def refine_statement(draft: str, feedback: str, word_limit: int) -> str:
    """Apply a single piece of feedback to the draft."""
    prompt = (
        f"You are editing a personal statement. Word limit: {word_limit} words.\n\n"
        f"CURRENT DRAFT ({count_words(draft)} words):\n{draft}\n\n"
        f"FEEDBACK: {feedback}\n\n"
        f"Apply only the requested changes. Stay within {word_limit} words. "
        f"Return the full revised statement only."
    )
    revised = chat(prompt, temperature=0.3)
    wc = count_words(revised)
    print(f"  Revised: {wc} words")
    return revised
