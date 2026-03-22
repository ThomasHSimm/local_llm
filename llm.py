# llm.py
from openai import OpenAI
from typing import Iterator

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "qwen2.5:7b"

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

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