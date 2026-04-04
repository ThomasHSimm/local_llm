"""
session.py
Save and load application sessions as JSON.
Enables resuming work across runs without re-running LLM calls.
"""

import json
from datetime import datetime
from pathlib import Path


def save_session(
    parsed_job: dict,
    parsed_cv: dict,
    gap_analysis: dict,
    questions: list,
    evidence: list,
    draft: str,
    word_limit: int,
    output_dir: str = ".",
) -> str:
    """Save the full session to a JSON file. Returns the file path."""
    org = (
        parsed_job.get("organisation", "application")
        .replace(" ", "_")
        .lower()[:20]
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"session_{org}_{timestamp}.json"
    path = Path(output_dir) / filename

    session = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "job_title": parsed_job.get("job_title", ""),
            "organisation": parsed_job.get("organisation", ""),
            "word_count": len(draft.split()),
            "word_limit": word_limit,
        },
        "parsed_job": parsed_job,
        "parsed_cv": parsed_cv,
        "gap_analysis": gap_analysis,
        "questions": questions,
        "evidence": evidence,
        "draft": draft,
        "word_limit": word_limit,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(session, f, indent=2)

    print(f"Session saved: {path}")
    return str(path)


def load_session(path: str) -> dict:
    """Load a session from a JSON file."""
    with open(path) as f:
        session = json.load(f)
    meta = session.get("metadata", {})
    print(
        f"Session loaded: {meta.get('job_title', '?')} "
        f"at {meta.get('organisation', '?')} "
        f"({meta.get('word_count', '?')}/{meta.get('word_limit', '?')} words)"
    )
    return session


def save_draft(draft: str, parsed_job: dict, word_limit: int, output_dir: str = ".") -> str:
    """Save the final draft as a plain text file."""
    org = (
        parsed_job.get("organisation", "application")
        .replace(" ", "_")
        .lower()[:20]
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"personal_statement_{org}_{timestamp}.txt"
    path = Path(output_dir) / filename

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"Personal Statement\n")
        f.write(f"{parsed_job.get('job_title', '')} — {parsed_job.get('organisation', '')}\n")
        f.write(f"Word count: {len(draft.split())}/{word_limit}\n")
        f.write("=" * 60 + "\n\n")
        f.write(draft)

    print(f"Draft saved: {path}")
    return str(path)
