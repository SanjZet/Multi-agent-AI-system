"""Retriever agent that answers questions from indexed context only."""

from __future__ import annotations

from dataclasses import dataclass

from core.llm import LLMCore
from core.vector_store import VectorStore


@dataclass
class RetrievalResult:
    """Result bundle returned by the retriever agent."""

    answer: str
    sources: list[dict]
    context_tokens_used: int
    insufficient_context: bool


class RetrieverAgent:
    """Performs retrieval-augmented answering over the vector store."""

    def __init__(self, vector_store: VectorStore, llm: LLMCore) -> None:
        """Initialize retriever with vector store and LLM core."""
        self.vector_store = vector_store
        self.llm = llm
        self.relevance_threshold = 0.65
        self.max_context_tokens = 3000

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        """Retrieve relevant chunks, build bounded context, and answer from context only."""
        hits = self.vector_store.hybrid_search(query, k=k)
        filtered = [hit for hit in hits if hit.get("score", 0.0) >= self.relevance_threshold]

        context_lines: list[str] = []
        context_tokens = 0
        for item in filtered:
            line = (
                f"[chunk_id={item.get('chunk_id', 'unknown')}] source={item.get('source', 'unknown')} "
                f"score={item.get('score', 0.0):.3f}\n{item.get('text', '')}"
            )
            line_tokens = self.llm.count_tokens(line, "gpt-3.5-turbo")
            if context_tokens + line_tokens > self.max_context_tokens:
                break
            context_lines.append(line)
            context_tokens += line_tokens

        context_text = "\n\n".join(context_lines)
        prompt = [
            {
                "role": "system",
                "content": (
                    "Answer the question using ONLY the provided context. "
                    "For each claim cite the chunk_id in [brackets]. "
                    "If the context doesn't contain the answer say: INSUFFICIENT_CONTEXT."
                ),
            },
            {"role": "user", "content": f"Question: {query}\n\nContext:\n{context_text or '(empty)'}"},
        ]

        answer = self.llm.chat(prompt)
        insufficient = "INSUFFICIENT_CONTEXT" in answer
        return RetrievalResult(
            answer=answer,
            sources=filtered,
            context_tokens_used=context_tokens,
            insufficient_context=insufficient,
        )

    def ingest_and_retrieve(self, text: str, query: str) -> str:
        """Ingest ad hoc text, then run immediate retrieval for a query."""
        self.vector_store.add_documents([{"text": text, "source": "ad_hoc", "metadata": {}}])
        result = self.retrieve(query)
        return result.answer
