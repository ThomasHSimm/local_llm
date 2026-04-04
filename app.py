"""
app.py
Streamlit UI for the Job Application Assistant.
Run with: streamlit run app.py

Requires: pip install streamlit
"""

import streamlit as st
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Job Application Assistant",
    page_icon="📝",
    layout="wide"
)

st.title("📝 Job Application Assistant")
st.caption("Analyse your fit, surface evidence, draft a personal statement.")

# ── Session state defaults ────────────────────────────────────────────────────
defaults = {
    "parsed_job": None,
    "parsed_cv": None,
    "gap_analysis": None,
    "questions": None,
    "evidence": None,
    "draft": None,
    "provider_ready": False,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    provider = st.selectbox(
        "LLM Provider",
        ["ollama", "groq", "anthropic", "openai"],
        help="Ollama = fully local, no data leaves your machine"
    )
    word_limit = st.number_input("Word limit", value=500, min_value=100, max_value=2000, step=50)
    n_questions = st.slider("Questions to generate", 3, 10, 6)

    if st.button("Connect to LLM"):
        try:
            from llm import use_provider
            use_provider(provider)
            st.session_state.provider_ready = True
            st.success(f"Connected: {provider}")
        except Exception as e:
            st.error(f"Connection failed: {e}")
            if provider == "ollama":
                st.caption("Make sure Ollama is running: `ollama serve`")

    st.divider()
    st.caption("🔒 With Ollama, your CV stays on your machine.")

# ── Warn if not connected ─────────────────────────────────────────────────────
if not st.session_state.provider_ready:
    st.info("Connect to an LLM provider in the sidebar to get started.")

# ── Stage 1: Ingest ───────────────────────────────────────────────────────────
st.header("1. Your Documents")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Job Specification")
    job_text = st.text_area(
        "Paste job spec",
        height=300,
        placeholder="Paste the full job specification text...",
        label_visibility="collapsed"
    )

with col2:
    st.subheader("Your CV")
    cv_text = st.text_area(
        "Paste CV",
        height=300,
        placeholder="Paste your CV text...",
        label_visibility="collapsed"
    )

# ── Stage 2: Analyse ──────────────────────────────────────────────────────────
st.header("2. Gap Analysis")

analyse_disabled = not (job_text and cv_text and st.session_state.provider_ready)

if st.button("▶ Run Analysis", disabled=analyse_disabled):
    with st.spinner("Parsing documents and running gap analysis..."):
        try:
            from job_assistant.analyser import (
                parse_job_spec_structured,
                parse_cv_structured,
                analyse_gaps,
            )
            st.session_state.parsed_job = parse_job_spec_structured(job_text)
            st.session_state.parsed_cv = parse_cv_structured(cv_text)
            st.session_state.gap_analysis = analyse_gaps(
                st.session_state.parsed_job,
                st.session_state.parsed_cv
            )
            # Reset downstream state
            st.session_state.questions = None
            st.session_state.evidence = None
            st.session_state.draft = None
        except Exception as e:
            st.error(f"Analysis failed: {e}")

if st.session_state.gap_analysis:
    gap = st.session_state.gap_analysis
    score = gap.get("overall_fit_score", "?")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fit Score", f"{score}/100")
    col2.metric("Strong matches", len(gap.get("strong_matches", [])))
    col3.metric("Partial matches", len(gap.get("partial_matches", [])))
    col4.metric("Missing", len(gap.get("missing", [])))

    st.info(gap.get("fit_summary", ""))

    with st.expander("Priority gaps to address"):
        for g in gap.get("priority_gaps", []):
            st.markdown(f"- {g}")

    with st.expander("Strong matches"):
        for m in gap.get("strong_matches", []):
            st.markdown(f"**{m.get('requirement', '')}**")
            st.caption(m.get("evidence", ""))

# ── Stage 3: Elicitation ──────────────────────────────────────────────────────
st.header("3. Your Evidence")
st.caption("Answer targeted questions to surface experience your CV doesn't capture.")

if st.session_state.gap_analysis:
    if st.button("▶ Generate Questions", disabled=st.session_state.questions is not None):
        with st.spinner("Generating questions..."):
            from job_assistant.elicitor import generate_questions
            st.session_state.questions = generate_questions(
                st.session_state.gap_analysis, n=n_questions
            )

if st.session_state.questions:
    st.markdown("Fill in as many answers as you can — more detail = better draft.")
    answers = {}
    for i, q in enumerate(st.session_state.questions, 1):
        with st.container():
            st.markdown(f"**Q{i}** — *{q.get('gap', '')}*")
            st.markdown(q.get("question", ""))
            if q.get("follow_up"):
                st.caption(f"If you're not sure: {q.get('follow_up', '')}")
            answers[i] = st.text_area(
                f"answer_{i}",
                height=80,
                key=f"answer_{i}",
                label_visibility="collapsed",
                placeholder="Your answer..."
            )
        st.divider()

# ── Stage 4: Draft ────────────────────────────────────────────────────────────
st.header("4. Personal Statement")

if st.session_state.questions:
    filled = {k: v for k, v in answers.items() if v and v.strip()}
    draft_disabled = len(filled) == 0

    if draft_disabled:
        st.caption("Fill in at least one answer above to enable drafting.")

    if st.button("▶ Draft Statement", disabled=draft_disabled):
        with st.spinner(f"Drafting ({word_limit} word target)..."):
            from job_assistant.elicitor import process_answers
            from job_assistant.writer import draft_statement
            st.session_state.evidence = process_answers(
                st.session_state.questions, filled
            )
            st.session_state.draft = draft_statement(
                st.session_state.parsed_job,
                st.session_state.parsed_cv,
                st.session_state.gap_analysis,
                st.session_state.evidence,
                word_limit=word_limit
            )

if st.session_state.draft:
    from job_assistant.writer import count_words
    wc = count_words(st.session_state.draft)

    col1, col2 = st.columns([1, 4])
    col1.metric("Words", f"{wc}/{word_limit}", delta=wc - word_limit, delta_color="inverse")

    edited = st.text_area(
        "Draft — edit directly or use refinement below",
        value=st.session_state.draft,
        height=400,
        key="draft_display"
    )

    # Update session state if user edits directly
    if edited != st.session_state.draft:
        st.session_state.draft = edited

    # ── Refinement ────────────────────────────────────────────────────────────
    st.subheader("Refine")
    feedback = st.text_input(
        "Feedback",
        placeholder="e.g. 'Strengthen the opening', 'Cut 50 words', 'Add more on the ethics work'"
    )

    if st.button("▶ Apply Feedback", disabled=not feedback):
        with st.spinner("Revising..."):
            from job_assistant.writer import refine_statement
            st.session_state.draft = refine_statement(
                st.session_state.draft, feedback, word_limit
            )
        st.rerun()

    # ── Save / Download ───────────────────────────────────────────────────────
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "⬇ Download draft (.txt)",
            data=st.session_state.draft,
            file_name="personal_statement.txt",
            mime="text/plain"
        )

    with col2:
        if st.button("💾 Save session"):
            from job_assistant.session import save_session
            import json
            path = save_session(
                st.session_state.parsed_job,
                st.session_state.parsed_cv,
                st.session_state.gap_analysis,
                st.session_state.questions,
                st.session_state.evidence,
                st.session_state.draft,
                word_limit,
                output_dir="outputs"
            )
            st.success(f"Saved: {path}")
