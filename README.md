# Job Application Assistant

An LLM-powered pipeline that analyses a job specification against your CV, identifies gaps, elicits evidence through targeted questions, and drafts a personal statement within a word limit.

Built on [Ollama](https://ollama.com) for fully local, private inference — your CV never leaves your machine. Swap to Groq, Anthropic or OpenAI with a one-line config change.

---

## Use case

Most job application tools generate a cover letter by summarising your CV against a job spec. This does something different: it conducts a structured **elicitation** — asking you targeted questions to surface evidence and examples that your CV doesn't capture — before drafting.

The result is a personal statement grounded in specific, accurate examples rather than generic claims. The quality depends on what you put into the answers, not on what the model invents.

---

## Example

The `example/` folder contains a complete worked example using a fictional candidate:

- `example/job_spec.txt` — a fictional Senior Data Scientist role at a transport analytics agency
- `example/cv.txt` — a fictional candidate (Sarah Chen, 5 years experience in transport and fintech DS)
- `example/answers_example.toml` — pre-written answers to representative elicitation questions
- `example/config_example.toml` — config wiring everything together

To run the example:

```bash
python job_assistant/main.py --config example/config_example.toml
```

### How it works — gap analysis and elicitation

Running the example produces a gap analysis first. On `qwen2.5:7b` this was:

```
Fit score: 56/100 (3 strong, 3 partial, 2 missing)
Summary: The candidate has strong matches for operational problem-solving,
real-world impact, and professional development, but partial matches for
deploying machine learning models, building reproducible pipelines, and
applying ethics frameworks. There are also gaps in experience with large
datasets and communicating findings to non-technical stakeholders.
```

Based on those gaps, the pipeline generates targeted questions. A few examples:

> **Q1** [Deploying ML models to solve operational problems]
> *Can you walk me through a specific project where you developed, deployed and evaluated a machine learning model to address an operational problem?*

> **Q3** [Applying ethics frameworks]
> *Can you describe a situation where you had to apply an ethics framework or data protection principle during your analysis, and how you ensured your work was compliant?*

> **Q5** [Communicating findings to non-technical stakeholders]
> *Can you give an example of a time when you had to communicate complex analytical findings to someone without a technical background, and how you approached it?*

The questions target the gaps — not a generic list, but driven by what the CV does and doesn't evidence. The answers to these questions (in `answers_example.toml`) are what the writer uses to build the draft.

### Example output

Run on `qwen2.5:7b` (a 4.7GB local model), the pipeline produced the following 474-word personal statement from Sarah Chen's CV and answers:

---

*As a Senior Data Scientist at Department for Transport Analytics (DfTA), I have a proven track record of using data science techniques to solve operational problems and drive real-world impact.*

*One notable example is my work on the road casualty risk model at West Yorkshire CA, where I reframed the problem from prioritising raw casualty counts to focusing on risk per vehicle kilometre. By building a gradient boosting model incorporating road geometry, traffic flow, and historical casualty data, I deployed it as a ranked corridor list that the safety team now uses to prioritise infrastructure work. This model has directly influenced three infrastructure projects in the last 18 months.*

*I have also successfully developed and deployed machine learning models to address operational challenges. At Lloyds, I deployed an XGBoost fraud scoring model into production via a FastAPI REST API on AWS ECS, processing real-time transactions at scale...*

*[474 words total — see `example/outputs/` for the full draft]*

---

### Model quality and elicitation

The example was run with `qwen2.5:7b`. The gap analysis identified 3 strong matches, 3 partial matches, and 2 gaps — some of which were false negatives (the model missed evidence that was in the CV). A larger model produces more accurate gap analysis and sharper elicitation questions:

| Model | Size | Gap analysis accuracy | Question quality | Speed (CPU) |
|---|---|---|---|---|
| `llama3.2:3b` | 2.0 GB | Low — misses evidence | Generic | Fast |
| `qwen2.5:7b` | 4.7 GB | Medium — some false gaps | Good | Moderate |
| `qwen2.5:14b` | 9.0 GB | High — reads CV carefully | Sharp | Slow |
| Cloud (Groq/Anthropic) | — | High | Sharp | Fast |

For development and testing, `qwen2.5:7b` is a good balance. For a real application, `qwen2.5:14b` or a cloud provider will produce noticeably better gap analysis and questions — which leads to better elicitation, which leads to a better draft.

The draft quality is ultimately determined by the answers you provide. A larger model elicits more specific questions; your answers to those questions are what the writer uses. Vague answers produce vague drafts regardless of model size.

---

## Installation

```bash
# Clone the repo
git clone https://github.com/ThomasHSimm/local_llm.git
cd local_llm

# Create environment
conda create -n llm_env python=3.13
conda activate llm_env

# Install dependencies
pip install openai python-dotenv requests pypdf python-docx tomli streamlit
```

Install and start Ollama:

```bash
# Install (Linux/Mac)
curl -fsSL https://ollama.com/install.sh | sh

# Start the server
ollama serve &

# Pull a model
ollama pull qwen2.5:7b    # recommended — 4.7GB, good quality
ollama pull llama3.2:3b   # faster, lower quality — good for testing
```

---

## Usage

### Step 1 — Set up your config

```bash
cp config_example.toml config.toml
```

Edit `config.toml`:
- Point `job_spec` and `cv` at your files (`.txt`, `.pdf`, or `.docx`)
- Set `provider = "ollama"` (or another provider)
- Set `answers_file` to a separate answers file, or leave blank and fill in `[answers]` inline

### Step 2 — Generate questions

```bash
python job_assistant/main.py --config config.toml --questions-only
```

Runs gap analysis and prints targeted questions. Questions are also saved to `outputs/questions.json`.

### Step 3 — Write your answers

Create an answers file (or fill in `[answers]` directly in `config.toml`):

```toml
# answers.toml
[answers]
1 = "Your answer to question 1 — aim for 50-150 words with specific examples"
2 = "Your answer to question 2"
```

Point your config at it:

```toml
[inputs]
answers_file = "answers.toml"
```

### Step 4 — Draft

```bash
python job_assistant/main.py --config config.toml
```

Saves the draft and full session to `outputs/`.

### Step 5 — Refine (optional)

Add refinement feedback to `config.toml`:

```toml
[refinements]
1 = "Strengthen the opening paragraph"
2 = "Cut 30 words from the second section"
```

Resume from the saved session:

```bash
python job_assistant/main.py --resume outputs/session_xxx.json --config config.toml
```

### Streamlit UI (optional)

```bash
streamlit run app.py
```

---

## Config reference

```toml
[inputs]
job_spec = "inputs/job_spec.txt"   # path to job spec (.txt, .pdf, .docx)
cv = "inputs/cv.txt"               # path to CV (.txt, .pdf, .docx)
answers_file = "answers.toml"      # optional — separate answers file

[settings]
provider = "ollama"                # ollama | groq | anthropic | openai
n_questions = 6                    # number of elicitation questions (3-10)
output_dir = "outputs"             # where to save sessions and drafts

[answers]
# Alternative to answers_file — fill in inline
# 1 = "Answer to question 1"

[refinements]
# Optional feedback applied to draft in order
# 1 = "First round of feedback"
```

---

## Switching providers

Change `provider` in `[settings]` and add your API key to a `.env` file:

```bash
# .env — never commit this
GROQ_API_KEY=gsk_...
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

| Provider | Model | Cost | Privacy |
|---|---|---|---|
| `ollama` | Local (qwen2.5:7b etc) | Free | Fully local |
| `groq` | Llama 3.1 70B | Free tier | Cloud |
| `anthropic` | Claude Haiku | ~$0.001/run | Cloud |
| `openai` | GPT-4o-mini | ~$0.001/run | Cloud |

**Privacy note:** With `ollama`, your CV and job spec never leave your machine. Use a cloud provider only if you are comfortable with that tradeoff.

---

## Project structure

```
local_llm/
├── llm.py                    # LLM client — swap provider with one variable
├── app.py                    # Streamlit UI
├── config_example.toml       # Config template
├── job_assistant/
│   ├── __init__.py
│   ├── parser.py             # File ingestion (txt, pdf, docx)
│   ├── analyser.py           # Structured extraction + gap analysis
│   ├── elicitor.py           # Question generation + answer processing
│   ├── writer.py             # PS drafting + word count enforcement
│   ├── session.py            # Save/load sessions
│   └── main.py               # CLI orchestrator
├── example/
│   ├── job_spec.txt          # Example job specification (fictional)
│   ├── cv.txt                # Example CV — Sarah Chen (fictional)
│   ├── answers_example.toml  # Example answers to representative questions
│   └── config_example.toml  # Example config
└── job_application_assistant.ipynb  # Kaggle-compatible notebook version
```

---

## Limitations

- Local 7B models occasionally produce malformed JSON — the pipeline retries automatically
- Gap analysis with smaller models may miss evidence that is clearly in the CV — use 14b or a cloud provider for more accurate analysis
- The draft is only as good as the answers provided — specific answers with concrete examples produce the best output
- Word counts are approximate — always verify the final output manually
- PDF parsing requires text-based PDFs; scanned PDFs need OCR (see `parser.py` comments)

---

## Related

- `local-llm-kaggle.ipynb` — general LLM playground notebook (Ollama, HuggingFace, fine-tuning)
- `app.py` — Streamlit UI wrapping the same pipeline