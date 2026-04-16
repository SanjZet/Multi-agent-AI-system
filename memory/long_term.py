"""Long-term memory backed by SQLite and semantic similarity recall."""

from __future__ import annotations

import pickle
import sqlite3
import hashlib
import re
from datetime import datetime

import numpy as np


class LongTermMemory:
    """Stores durable memory entries and recalls them using embedding similarity."""

    def __init__(self, db_path: str) -> None:
        """Initialize DB schema and embedding model."""
        self.db_path = db_path
        self.model = self._load_embedding_model()
        self.embedding_dim = self.model.get_sentence_embedding_dimension() if self.model else 384
        self._init_db()

    def _load_embedding_model(self):
        """Try to load sentence-transformers and fall back to hashing."""
        try:
            from sentence_transformers import SentenceTransformer

            return SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            return None

    def _embed_text(self, text: str) -> np.ndarray:
        """Embed text with the active embedding backend."""
        if self.model is not None:
            return self.model.encode(text, convert_to_numpy=True).astype("float32")

        vector = np.zeros(self.embedding_dim, dtype="float32")
        for token in re.findall(r"\w+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.embedding_dim
            vector[index] += 1.0
        return vector

    def _init_db(self) -> None:
        """Create long-term memory table if needed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    tags TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def store(self, text: str, tags: list[str] | None = None) -> None:
        """Store a text memory with optional tags."""
        used_tags = tags or []
        embedding = self._embed_text(text)
        payload = pickle.dumps(embedding)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO memories (text, embedding, tags, created_at) VALUES (?, ?, ?, ?)",
                (text, payload, ",".join(used_tags), datetime.utcnow().isoformat()),
            )
            conn.commit()

    def recall(self, query: str, k: int = 3) -> list[dict]:
        """Recall top-k memories by cosine similarity to the query."""
        q = self._embed_text(query)
        q_norm = np.linalg.norm(q) or 1.0

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT id, text, embedding, tags FROM memories").fetchall()

        scored: list[dict] = []
        for mem_id, text, embedding_blob, tags in rows:
            emb = pickle.loads(embedding_blob)
            emb_norm = np.linalg.norm(emb) or 1.0
            similarity = float(np.dot(q, emb) / (q_norm * emb_norm))
            scored.append(
                {
                    "id": int(mem_id),
                    "text": text,
                    "tags": [tag for tag in tags.split(",") if tag],
                    "similarity": similarity,
                }
            )

        scored.sort(key=lambda item: item["similarity"], reverse=True)
        return scored[:k]

    def delete(self, memory_id: int) -> None:
        """Delete a memory record by id."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()

    def to_prompt_context(self, query: str) -> str:
        """Format recalled memories into a prompt-friendly context block."""
        recalled = self.recall(query, k=3)
        if not recalled:
            return "No relevant long-term memories found."

        lines = ["Relevant long-term memories:"]
        for item in recalled:
            tags = ", ".join(item["tags"]) or "none"
            lines.append(
                f"- [Memory #{item['id']}] similarity={item['similarity']:.3f}, tags={tags}: {item['text']}"
            )
        return "\n".join(lines)
