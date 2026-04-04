"""
elicitor.py
Generate targeted questions based on gap analysis,
then process answers provided via config file.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm import chat


def generate_questions(gap_analysis: dict, n: int = 6) -> list:
    """
    Generate targeted elicitation questions based on gap analysis.
    Returns a list of question objects with gap, question, why, follow_up.
    """
    print(f"Generating {n} elicitation questions...")

    system = (
        "You return ONLY a valid JSON array. "
        "No explanation, no markdown, no extra text."
    )

    schema = json.dumps([{
        "gap": "the requirement being addressed",
        "question": "the specific question to ask the candidate",
        "why": "what evidence this question is trying to surface",
        "follow_up": "a follow-up if the answer is vague"
    }])

    prompt = (
        f"Based on this gap analysis, generate {n} targeted questions to elicit "
        f"evidence the CV doesn't capture. Focus on priority gaps and partial matches.\n\n"
        f"GAP ANALYSIS:\n{json.dumps(gap_analysis, indent=2)}\n\n"
        f"Questions should be concrete, open-ended, and ordered most to least important.\n"
        f"Return a JSON array matching this schema: {schema}"
    )

    raw = chat(prompt, system=system, temperature=0.2)

    try:
        text = raw.strip().strip("```json").strip("```").strip()
        if not text.startswith("["):
            text = text[text.find("["):text.rfind("]") + 1]
        questions = json.loads(text)

        # Normalise — model sometimes returns plain strings instead of objects
        normalised = []
        for q in questions:
            if isinstance(q, str):
                normalised.append({
                    "gap": "general requirement",
                    "question": q,
                    "why": "",
                    "follow_up": "Can you give a specific example?"
                })
            elif isinstance(q, dict):
                normalised.append(q)

        print(f"  {len(normalised)} questions generated")
        return normalised

    except Exception as e:
        print(f"  Failed to parse questions: {e}")
        return []


def print_questions(questions: list):
    """Print questions to stdout so the user can fill in their config."""
    print("\n" + "=" * 60)
    print("QUESTIONS — copy into config.toml under [answers]")
    print("=" * 60)
    for i, q in enumerate(questions, 1):
        print(f"\nQ{i} [{q.get('gap', '')}]")
        print(f"   {q.get('question', '')}")
        if q.get("follow_up"):
            print(f"   Follow-up if vague: {q.get('follow_up', '')}")
    print()


def process_answers(questions: list, answers: dict) -> list:
    """
    Match provided answers to questions.
    answers: dict mapping question number (int) to answer string.
    Returns list of evidence dicts ready for the writer.
    """
    collected = []
    for i, q in enumerate(questions, 1):
        answer = answers.get(i, answers.get(str(i), "")).strip()
        if not answer:
            print(f"  Q{i}: skipped")
            continue
        collected.append({
            "gap": q.get("gap", ""),
            "question": q.get("question", ""),
            "answer": answer
        })
        print(f"  Q{i}: loaded ({len(answer.split())} words)")
    print(f"  {len(collected)} answers collected")
    return collected
