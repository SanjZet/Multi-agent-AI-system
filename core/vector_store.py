"""Hybrid vector and keyword retrieval storage using FAISS and BM25."""

from __future__ import annotations

import logging
import os
import pickle
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import tiktoken
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


@dataclass
class IndexedDocument:
    """A single indexed chunk entry."""

    text: str
    source: str
    metadata: dict
    chunk_id: str


class VectorStore:
    """Persistent vector store with FAISS, BM25, hybrid ranking, and MMR."""

    def __init__(self, persist_path: str) -> None:
        """Load or create a persistent FAISS index and metadata store."""
        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)

        self.model = self._load_embedding_model()
        self.embedding_dim = self.model.get_sentence_embedding_dimension() if self.model else 384
        self.index_path = self.persist_path / "index.faiss"
        self.docs_path = self.persist_path / "documents.pkl"
        self.documents: list[dict] = []
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.load()

    def _load_embedding_model(self):
        """Try to load sentence-transformers and fall back to hash embeddings."""
        try:
            from sentence_transformers import SentenceTransformer

            return SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("sentence-transformers unavailable, using fallback embeddings: %s", exc)
            return None

    def _embed_text(self, text: str) -> np.ndarray:
        """Embed text using the active embedding backend."""
        if self.model is not None:
            vector = self.model.encode(text, convert_to_numpy=True).astype("float32")
            return self._normalize(vector)

        vector = np.zeros(self.embedding_dim, dtype="float32")
        for token in re.findall(r"\w+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.embedding_dim
            vector[index] += 1.0
        return self._normalize(vector)

    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
        """Split text into token-aware overlapping chunks."""
        tokens = self.encoding.encode(text or "")
        if not tokens:
            return []

        chunks: list[str] = []
        step = max(1, chunk_size - overlap)
        for start in range(0, len(tokens), step):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            if not chunk_tokens:
                continue
            chunks.append(self.encoding.decode(chunk_tokens))
            if end >= len(tokens):
                break
        return chunks

    def add_documents(self, docs: list[dict]) -> int:
        """Chunk, embed, and add documents to FAISS and BM25 stores."""
        embeddings: list[np.ndarray] = []
        added = 0

        for doc_idx, doc in enumerate(docs):
            text = doc.get("text", "")
            source = doc.get("source", "unknown")
            metadata = doc.get("metadata", {})
            chunks = self.chunk_text(text)
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"{len(self.documents)}_{doc_idx}_{chunk_idx}"
                embedding = self._embed_text(chunk)
                embeddings.append(embedding)
                self.documents.append(
                    {
                        "text": chunk,
                        "source": source,
                        "metadata": metadata,
                        "chunk_id": chunk_id,
                    }
                )
                added += 1

        if embeddings:
            matrix = np.vstack(embeddings).astype("float32")
            self.index.add(matrix)
            self.persist()

        return added

    def faiss_search(self, query: str, k: int) -> list[dict]:
        """Search by semantic similarity with cosine similarity scores."""
        if self.index.ntotal == 0 or not self.documents:
            return []

        query_vec = self._embed_text(query)
        scores, indices = self.index.search(np.expand_dims(query_vec, axis=0), k)

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            results.append(
                {
                    "text": doc["text"],
                    "source": doc["source"],
                    "metadata": doc["metadata"],
                    "chunk_id": doc["chunk_id"],
                    "score": float(score),
                    "search_type": "faiss",
                }
            )
        return results

    def bm25_search(self, query: str, k: int) -> list[dict]:
        """Search by keyword relevance using BM25 ranking."""
        if not self.documents:
            return []

        corpus_tokens = [doc["text"].lower().split() for doc in self.documents]
        bm25 = BM25Okapi(corpus_tokens)
        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)

        top_indices = np.argsort(scores)[::-1][:k]
        max_score = float(np.max(scores)) if len(scores) > 0 else 1.0
        max_score = max(max_score, 1e-9)

        results: list[dict] = []
        for idx in top_indices:
            raw_score = float(scores[idx])
            normalized = raw_score / max_score
            doc = self.documents[int(idx)]
            results.append(
                {
                    "text": doc["text"],
                    "source": doc["source"],
                    "metadata": doc["metadata"],
                    "chunk_id": doc["chunk_id"],
                    "score": normalized,
                    "search_type": "bm25",
                }
            )
        return results

    def hybrid_search(self, query: str, k: int = 5, faiss_weight: float = 0.6) -> list[dict]:
        """Blend FAISS and BM25 signals, then apply MMR and score filtering."""
        faiss_results = self.faiss_search(query, max(k * 3, 10))
        bm25_results = self.bm25_search(query, max(k * 3, 10))

        merged: dict[str, dict] = {}
        for result in faiss_results:
            merged[result["chunk_id"]] = {**result, "combined_score": faiss_weight * result["score"]}

        for result in bm25_results:
            existing = merged.get(result["chunk_id"])
            bm25_component = (1.0 - faiss_weight) * result["score"]
            if existing:
                existing["combined_score"] += bm25_component
            else:
                merged[result["chunk_id"]] = {**result, "combined_score": bm25_component}

        ranked = sorted(merged.values(), key=lambda x: x["combined_score"], reverse=True)
        mmr_selected = self._apply_mmr(query, ranked, top_k=k * 2)
        filtered = [r for r in mmr_selected if r["combined_score"] >= 0.45]

        return [
            {
                "text": item["text"],
                "source": item["source"],
                "metadata": item["metadata"],
                "chunk_id": item["chunk_id"],
                "score": float(item["combined_score"]),
            }
            for item in filtered[:k]
        ]

    def _apply_mmr(self, query: str, candidates: list[dict], top_k: int, lambda_param: float = 0.7) -> list[dict]:
        """Apply Maximal Marginal Relevance to reduce redundancy."""
        if not candidates:
            return []

        query_vec = self._embed_text(query)
        candidate_vecs = [self._embed_text(item["text"]) for item in candidates]

        selected_indices: list[int] = []
        available = set(range(len(candidates)))

        while available and len(selected_indices) < top_k:
            if not selected_indices:
                best_idx = max(available, key=lambda i: candidates[i]["combined_score"])
                selected_indices.append(best_idx)
                available.remove(best_idx)
                continue

            best_idx = None
            best_score = -1e9
            for idx in available:
                relevance = float(np.dot(candidate_vecs[idx], query_vec))
                diversity = max(float(np.dot(candidate_vecs[idx], candidate_vecs[j])) for j in selected_indices)
                mmr_score = lambda_param * relevance - (1.0 - lambda_param) * diversity
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is None:
                break
            selected_indices.append(best_idx)
            available.remove(best_idx)

        return [candidates[i] for i in selected_indices]

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalize a vector for cosine similarity search."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def persist(self) -> None:
        """Save FAISS index and document metadata to disk."""
        faiss.write_index(self.index, str(self.index_path))
        with self.docs_path.open("wb") as fp:
            pickle.dump(self.documents, fp)

    def load(self) -> None:
        """Load FAISS index and documents from disk when available."""
        if self.index_path.exists() and self.docs_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                with self.docs_path.open("rb") as fp:
                    self.documents = pickle.load(fp)
            except (OSError, pickle.UnpicklingError, RuntimeError) as exc:
                logger.warning("Failed to load vector store, starting fresh: %s", exc)
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                self.documents = []
        else:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.documents = []
