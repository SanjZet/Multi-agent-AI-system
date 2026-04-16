"""YouTube ingestion utilities for metadata extraction, audio download, and transcription."""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import time
from datetime import datetime
from pathlib import Path

from yt_dlp import YoutubeDL

from core.llm import LLMCore

logger = logging.getLogger(__name__)


class YouTubeTool:
    """Utilities for downloading and analyzing YouTube content."""

    def __init__(self) -> None:
        """Initialize reusable Whisper model state lazily."""
        self._whisper_model = None

    def _log_tool_call(self, tool_name: str, payload: dict, success: bool, latency_ms: float) -> None:
        """Emit standardized tool call logs."""
        input_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        logger.info(
            "tool_call timestamp=%s tool_name=%s input_hash=%s success=%s latency_ms=%.2f",
            datetime.utcnow().isoformat(),
            tool_name,
            input_hash,
            success,
            latency_ms,
        )

    def _get_whisper_model(self):
        """Load and cache the Whisper base model."""
        if self._whisper_model is None:
            try:
                import whisper
            except ImportError as exc:
                raise RuntimeError(
                    "Whisper is not installed in the current environment. "
                    "Install a compatible openai-whisper build or use a supported Python version."
                ) from exc
            self._whisper_model = whisper.load_model("base")
        return self._whisper_model

    def download_audio(self, url: str) -> str:
        """Download YouTube audio as mp3 and return local file path."""
        start = time.perf_counter()
        payload = {"url": url}
        try:
            tmp_dir = Path(tempfile.gettempdir())
            output_template = str(tmp_dir / "%(id)s.%(ext)s")
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": output_template,
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "192",
                    }
                ],
                "quiet": True,
                "noprogress": True,
            }

            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                file_path = tmp_dir / f"{info['id']}.mp3"

            latency = (time.perf_counter() - start) * 1000
            self._log_tool_call("youtube_download_audio", payload, True, latency)
            return str(file_path)
        except (KeyError, FileNotFoundError, OSError) as exc:
            latency = (time.perf_counter() - start) * 1000
            self._log_tool_call("youtube_download_audio", payload, False, latency)
            logger.exception("Failed to download audio")
            raise RuntimeError(f"YouTube audio download failed: {exc}") from exc

    def transcribe(self, audio_path: str) -> list[dict]:
        """Transcribe audio using Whisper base model and return segmented text."""
        start = time.perf_counter()
        payload = {"audio_path": audio_path}
        try:
            model = self._get_whisper_model()
            result = model.transcribe(audio_path)
            segments = [
                {
                    "text": seg.get("text", "").strip(),
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", 0.0)),
                }
                for seg in result.get("segments", [])
            ]
            latency = (time.perf_counter() - start) * 1000
            self._log_tool_call("youtube_transcribe", payload, True, latency)
            return segments
        except (FileNotFoundError, RuntimeError, OSError) as exc:
            latency = (time.perf_counter() - start) * 1000
            self._log_tool_call("youtube_transcribe", payload, False, latency)
            logger.exception("Whisper transcription failed")
            raise RuntimeError(f"Transcription failed: {exc}") from exc

    def extract_code_blocks(self, transcript_text: str, llm: LLMCore) -> list[str]:
        """Use LLM to extract explicit or implied code snippets from transcript text."""
        start = time.perf_counter()
        payload = {"transcript_preview": transcript_text[:500]}
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Extract all code snippets mentioned or described in the transcript. "
                        "Return strict JSON: {\"code_blocks\": [\"...\"]}."
                    ),
                },
                {"role": "user", "content": transcript_text[:12000]},
            ]
            raw = llm.chat(messages)
            parsed = json.loads(raw)
            code_blocks = [str(item) for item in parsed.get("code_blocks", []) if str(item).strip()]
            latency = (time.perf_counter() - start) * 1000
            self._log_tool_call("youtube_extract_code_blocks", payload, True, latency)
            return code_blocks
        except (json.JSONDecodeError, ValueError, RuntimeError) as exc:
            latency = (time.perf_counter() - start) * 1000
            self._log_tool_call("youtube_extract_code_blocks", payload, False, latency)
            logger.exception("Code block extraction failed")
            return []

    def get_metadata(self, url: str) -> dict:
        """Fetch basic video metadata without downloading content."""
        start = time.perf_counter()
        payload = {"url": url}
        try:
            ydl_opts = {"quiet": True, "noprogress": True, "skip_download": True}
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

            metadata = {
                "title": info.get("title", ""),
                "channel": info.get("uploader", ""),
                "duration": info.get("duration", 0),
                "upload_date": info.get("upload_date", ""),
                "description": info.get("description", ""),
            }
            latency = (time.perf_counter() - start) * 1000
            self._log_tool_call("youtube_get_metadata", payload, True, latency)
            return metadata
        except (KeyError, RuntimeError, OSError) as exc:
            latency = (time.perf_counter() - start) * 1000
            self._log_tool_call("youtube_get_metadata", payload, False, latency)
            logger.exception("Metadata retrieval failed")
            raise RuntimeError(f"Metadata fetch failed: {exc}") from exc
