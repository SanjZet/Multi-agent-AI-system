"""Specialized agent for converting YouTube videos into structured study artifacts."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from core.llm import LLMCore
from core.vector_store import VectorStore
from tools.youtube_tool import YouTubeTool


@dataclass
class YouTubeReport:
    """Structured output of YouTube processing."""

    url: str
    title: str
    transcript_chunks: int
    notes_markdown: str
    code_files: list[str]
    flashcards: list[dict]
    processing_time_seconds: float


class YouTubeAgent:
    """Processes YouTube content into notes, embeddings, code files, and flashcards."""

    def __init__(self, youtube_tool: YouTubeTool, vector_store: VectorStore, llm: LLMCore) -> None:
        """Initialize YouTube agent dependencies."""
        self.youtube_tool = youtube_tool
        self.vector_store = vector_store
        self.llm = llm

    def process(self, url: str) -> YouTubeReport:
        """Run full YouTube ingestion and report generation pipeline."""
        started = time.perf_counter()
        metadata = self.youtube_tool.get_metadata(url)
        audio_path = self.youtube_tool.download_audio(url)
        segments = self.youtube_tool.transcribe(audio_path)

        docs = []
        for idx, segment in enumerate(segments):
            docs.append(
                {
                    "text": segment["text"],
                    "source": url,
                    "metadata": {
                        "source": url,
                        "type": "youtube",
                        "title": metadata.get("title", ""),
                        "timestamp": f"{segment['start']:.2f}-{segment['end']:.2f}",
                        "segment_index": idx,
                    },
                }
            )
        self.vector_store.add_documents(docs)

        transcript_text = "\n".join(seg["text"] for seg in segments)
        code_snippets = self.youtube_tool.extract_code_blocks(transcript_text, self.llm)

        prompt = [
            {
                "role": "system",
                "content": (
                    "Generate structured markdown notes with sections: Overview (3 sentences), "
                    "Key Concepts (bullet list), Code Examples (runnable Python), Action Items, "
                    "Study Flashcards (10 Q&A pairs as JSON)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Title: {metadata.get('title', '')}\nURL: {url}\n\n"
                    f"Transcript:\n{transcript_text[:20000]}\n\n"
                    f"Extracted code hints:\n{code_snippets}"
                ),
            },
        ]
        notes_markdown = self.llm.chat(prompt)

        flashcards = self._extract_flashcards(notes_markdown)
        code_files = self._save_code_files(code_snippets)

        elapsed = time.perf_counter() - started
        return YouTubeReport(
            url=url,
            title=metadata.get("title", ""),
            transcript_chunks=len(segments),
            notes_markdown=notes_markdown,
            code_files=code_files,
            flashcards=flashcards,
            processing_time_seconds=float(round(elapsed, 2)),
        )

    def _extract_flashcards(self, notes_markdown: str) -> list[dict]:
        """Extract flashcards JSON array from generated markdown."""
        start = notes_markdown.find("[")
        end = notes_markdown.rfind("]")
        if start >= 0 and end > start:
            try:
                data = json.loads(notes_markdown[start : end + 1])
                if isinstance(data, list):
                    return [item for item in data if isinstance(item, dict)]
            except json.JSONDecodeError:
                return []
        return []

    def _save_code_files(self, snippets: list[str]) -> list[str]:
        """Persist extracted code snippets into output Python files."""
        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []

        for idx, snippet in enumerate(snippets, start=1):
            file_path = output_dir / f"youtube_snippet_{idx}.py"
            with file_path.open("w", encoding="utf-8") as fp:
                fp.write(snippet.strip() + "\n")
            paths.append(str(file_path))

        return paths
