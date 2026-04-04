# llm.py
import os
import json
from openai import OpenAI
from typing import Iterator

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "qwen2.5:7b"

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# ── Core chat functions ───────────────────────────────────────────────────────


def chat(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system: str = None,
    temperature: float = 0.7,
    history: list = None,
) -> str:
    """Single-turn chat. Returns the response as a string."""
    messages = _build_messages(prompt, system, history)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


def stream(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system: str = None,
    temperature: float = 0.7,
) -> Iterator[str]:
    """Stream tokens as they're generated. Use in a for loop."""
    messages = _build_messages(prompt, system)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    for chunk in response:
        token = chunk.choices[0].delta.content
        if token:
            yield token


def stream_print(prompt: str, model: str = DEFAULT_MODEL, system: str = None):
    """Stream directly to stdout — convenience wrapper."""
    for token in stream(prompt, model=model, system=system):
        print(token, end="", flush=True)
    print()


# ── Conversation class ────────────────────────────────────────────────────────


class Conversation:
    """Multi-turn chat with memory."""

    def __init__(self, model: str = DEFAULT_MODEL, system: str = None):
        self.model = model
        self.history = []
        if system:
            self.history.append({"role": "system", "content": system})

    def say(self, message: str, stream: bool = False) -> str:
        self.history.append({"role": "user", "content": message})

        if stream:
            reply = self._stream_reply()
        else:
            reply = self._reply()

        self.history.append({"role": "assistant", "content": reply})
        return reply

    def _reply(self) -> str:
        response = client.chat.completions.create(
            model=self.model,
            messages=self.history,
        )
        return response.choices[0].message.content

    def _stream_reply(self) -> str:
        response = client.chat.completions.create(
            model=self.model,
            messages=self.history,
            stream=True,
        )
        tokens = []
        for chunk in response:
            token = chunk.choices[0].delta.content
            if token:
                print(token, end="", flush=True)
                tokens.append(token)
        print()
        return "".join(tokens)

    def reset(self):
        """Clear history but keep system prompt."""
        self.history = [m for m in self.history if m["role"] == "system"]

    def __len__(self):
        return len([m for m in self.history if m["role"] == "user"])


# ── Utility ───────────────────────────────────────────────────────────────────


def models() -> list[str]:
    """List all locally available models."""
    return [m.id for m in client.models.list().data]


def _build_messages(prompt, system=None, history=None) -> list:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    return messages


# Helper to switch provider at runtime (useful in notebooks)
def use_provider(provider: str = None):
    global client, DEFAULT_MODEL
    if provider is None:
        provider = "ollama"  # "ollama" | "openai" | "anthropic" | "groq" | "runpod"

    PROVIDERS = {
        "ollama": {
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
            "model": "qwen2.5:7b",
        },
        "openai": {
            "base_url": None,  # uses OpenAI default
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": "gpt-4o-mini",  # cheapest capable model
        },
        "anthropic": {
            "base_url": "https://api.anthropic.com/v1",  # via openai-compat layer
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model": "claude-haiku-4-5-20251001",  # cheapest Claude
        },
        "groq": {
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": os.getenv("GROQ_API_KEY"),
            "model": "llama-3.1-70b-versatile",  # fast & free tier available
        },
    }
    cfg = PROVIDERS[provider]
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
    DEFAULT_MODEL = cfg["model"]
    print(f"✅ Switched to {provider} — default model: {DEFAULT_MODEL}")


def extract_json(prompt: str, schema: str, model: str = DEFAULT_MODEL) -> dict:
    """Ask the model to return structured JSON."""
    system = "You return only valid JSON. No explanation, no markdown, no backticks."
    full_prompt = f"{prompt}\n\nReturn this exact schema: {schema}"
    raw = chat(full_prompt, system=system, model=model, temperature=0.1)
    return json.loads(raw.strip())


def classify(text: str, labels: list[str], model: str = DEFAULT_MODEL) -> str:
    """Classify text into one of the provided labels."""
    system = f"Classify the input. Reply with ONLY one of these labels, nothing else: {', '.join(labels)}"
    return chat(text, system=system, model=model, temperature=0.0).strip()


def transform(text: str, instruction: str, model: str = DEFAULT_MODEL) -> str:
    """Apply a transformation to text."""
    return chat(f"{instruction}:\n\n{text}", temperature=0.3, model=model)


def summarise(text: str, style: str = "bullet", max_points: int = 5) -> str:
    styles = {
        "bullet": f"Summarise in up to {max_points} bullet points.",
        "one_line": "Summarise in exactly one sentence.",
        "eli5": "Explain like I'm 5 years old.",
        "technical": f"Write a technical summary in up to {max_points} points.",
    }
    return chat(f"{styles[style]}\n\n{text}", temperature=0.3)


def parse_json(raw: str) -> dict:
    """Robustly extract JSON from a model response."""
    if not raw or not raw.strip():
        raise ValueError("Model returned an empty response")

    text = raw.strip()

    # Strip markdown code fences if present
    # e.g. ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        text = "\n".join(lines[1:-1]).strip()

    # If model still wrapped it, find the first { and last }
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]

    return json.loads(text)
