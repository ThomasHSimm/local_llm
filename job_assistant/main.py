"""
main.py
Orchestrates the full job application assistant pipeline.

Usage:
    # Step 1 — generate questions (no answers needed yet)
    python job_assistant/main.py --config config.toml --questions-only

    # Step 2 — fill in [answers] in config.toml, then run full pipeline
    python job_assistant/main.py --config config.toml

    # Step 3 — resume from a saved session and apply new refinements
    python job_assistant/main.py --resume outputs/session_dvsa_20260401.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tomllib          # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # pip install tomli for older Python
    except ImportError:
        print("Install tomli: pip install tomli")
        sys.exit(1)

from llm import use_provider
from job_assistant.parser import parse_job_spec, parse_cv
from job_assistant.analyser import (
    parse_job_spec_structured,
    parse_cv_structured,
    analyse_gaps,
)
from job_assistant.elicitor import generate_questions, print_questions, process_answers
from job_assistant.writer import draft_statement, refine_statement, count_words
from job_assistant.session import save_session, load_session, save_draft


def load_config(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def run_pipeline(config: dict, questions_only: bool = False):
    """Run the full pipeline from config."""

    # ── Setup ─────────────────────────────────────────────────────────────────
    settings = config.get("settings", {})
    provider = settings.get("provider", "ollama")
    use_provider(provider)
    print(f"Provider: {provider}\n")

    output_dir = settings.get("output_dir", "outputs")
    n_questions = settings.get("n_questions", 6)

    # ── Parse inputs ──────────────────────────────────────────────────────────
    inputs = config.get("inputs", {})
    job_text = parse_job_spec(inputs.get("job_spec", ""))
    cv_text = parse_cv(inputs.get("cv", ""))

    if not job_text.strip() or not cv_text.strip():
        print("Job spec or CV is empty — check your config [inputs] section")
        sys.exit(1)

    # ── Structured extraction ─────────────────────────────────────────────────
    parsed_job = parse_job_spec_structured(job_text)
    parsed_cv = parse_cv_structured(cv_text)
    word_limit = parsed_job.get("word_limit") or 500

    # ── Gap analysis ──────────────────────────────────────────────────────────
    gap_analysis = analyse_gaps(parsed_job, parsed_cv, cv_text=cv_text)

    print(f"\nFit score: {gap_analysis.get('overall_fit_score', '?')}/100")
    print(f"Summary: {gap_analysis.get('fit_summary', '')}\n")

    # ── Generate questions ────────────────────────────────────────────────────
    questions = generate_questions(gap_analysis, n=n_questions)
    print_questions(questions)

    if questions_only:
        q_path = Path(output_dir) / "questions.json"
        q_path.parent.mkdir(parents=True, exist_ok=True)
        with open(q_path, "w") as f:
            json.dump(questions, f, indent=2)
        print(f"Questions saved to {q_path}")
        print("Fill in [answers] in your config.toml then re-run without --questions-only")
        return

    # ── Process answers ───────────────────────────────────────────────────────
    # Load from separate answers_file if specified, else fall back to [answers] in config
    answers_file = config.get("inputs", {}).get("answers_file", "")
    if answers_file and Path(answers_file).exists():
        with open(answers_file, "rb") as f:
            answers_config = tomllib.load(f)
        raw_answers = answers_config.get("answers", {})
        print(f"Answers loaded from: {answers_file}")
    else:
        raw_answers = config.get("answers", {})

    answers = {int(k): v for k, v in raw_answers.items()}

    if not answers:
        print("No answers found.")
        print("Either set answers_file in [inputs] or fill in [answers] in your config.")
        print("Run with --questions-only first to see the questions.")
        sys.exit(0)

    evidence = process_answers(questions, answers)

    # ── Draft ─────────────────────────────────────────────────────────────────
    print()
    draft = draft_statement(parsed_job, parsed_cv, gap_analysis, evidence, word_limit)

    # ── Apply refinements from config ─────────────────────────────────────────
    refinements = config.get("refinements", {})
    if refinements:
        print(f"\nApplying {len(refinements)} refinement(s)...")
        for i in sorted(refinements.keys()):
            feedback = refinements[i]
            print(f"  Refinement {i}: {feedback[:60]}...")
            draft = refine_statement(draft, feedback, word_limit)

    # ── Print final draft ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"FINAL DRAFT — {count_words(draft)}/{word_limit} words")
    print("=" * 60)
    print(draft)
    print("=" * 60)

    # ── Save outputs ──────────────────────────────────────────────────────────
    save_session(
        parsed_job, parsed_cv, gap_analysis,
        questions, evidence, draft, word_limit,
        output_dir=output_dir
    )
    save_draft(draft, parsed_job, word_limit, output_dir=output_dir)


def resume_pipeline(session_path: str, config: dict = None):
    """Resume from a saved session, optionally applying new refinements."""
    session = load_session(session_path)

    draft = session["draft"]
    word_limit = session["word_limit"]
    parsed_job = session["parsed_job"]

    print(f"\nCurrent draft: {count_words(draft)}/{word_limit} words\n")
    print(draft)
    print()

    if config:
        refinements = config.get("refinements", {})
        if refinements:
            print(f"Applying {len(refinements)} refinement(s)...")
            use_provider(config.get("settings", {}).get("provider", "ollama"))
            for i in sorted(refinements.keys()):
                feedback = refinements[i]
                print(f"  Refinement {i}: {feedback[:60]}...")
                draft = refine_statement(draft, feedback, word_limit)

            print("\n" + "=" * 60)
            print(f"REVISED DRAFT — {count_words(draft)}/{word_limit} words")
            print("=" * 60)
            print(draft)
            print("=" * 60)

            output_dir = config.get("settings", {}).get("output_dir", "outputs")
            save_draft(draft, parsed_job, word_limit, output_dir=output_dir)


def main():
    parser = argparse.ArgumentParser(description="Job Application Assistant")
    parser.add_argument("--config", default="config.toml", help="Path to config file")
    parser.add_argument(
        "--questions-only", action="store_true",
        help="Generate questions only, do not draft"
    )
    parser.add_argument("--resume", help="Resume from a saved session JSON file")
    args = parser.parse_args()

    config = {}
    if Path(args.config).exists():
        config = load_config(args.config)
    elif not args.resume:
        print(f"Config file not found: {args.config}")
        print("Copy config_example.toml to config.toml and fill in your details")
        sys.exit(1)

    if args.resume:
        resume_pipeline(args.resume, config if config else None)
    else:
        run_pipeline(config, questions_only=args.questions_only)


if __name__ == "__main__":
    main()
