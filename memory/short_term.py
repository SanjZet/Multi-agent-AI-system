"""Short-term conversational memory with automatic summarization."""

from __future__ import annotations

from datetime import datetime

from core.llm import LLMCore


class ShortTermMemory:
    """Maintains a rolling message window and summarizes older turns when full."""

    def __init__(self, llm: LLMCore, max_messages: int = 20) -> None:
        """Initialize short-term memory with an LLM summarizer."""
        self.max_messages = max_messages
        self.llm = llm
        self.messages: list[dict] = []

    def add(self, role: str, content: str) -> None:
        """Append a message and summarize older context when capacity is exceeded."""
        self.messages.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        self.summarize_if_full()

    def get_window(self, n: int = 10) -> list[dict]:
        """Return the latest n messages."""
        return self.messages[-n:]

    def summarize_if_full(self) -> None:
        """Compress the oldest 10 messages into one system summary when over capacity."""
        if len(self.messages) <= self.max_messages:
            return

        oldest = self.messages[:10]
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in oldest)
        prompt = [
            {
                "role": "system",
                "content": "Summarize the following conversation snippets preserving facts, constraints, and decisions.",
            },
            {"role": "user", "content": transcript},
        ]
        summary = self.llm.chat(prompt)
        summary_msg = {
            "role": "system",
            "content": f"Conversation summary: {summary}",
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.messages = [summary_msg] + self.messages[10:]

    def to_prompt_context(self) -> list[dict[str, str]]:
        """Return messages formatted for chat APIs."""
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]

    def clear(self) -> None:
        """Clear all stored short-term messages."""
        self.messages.clear()
