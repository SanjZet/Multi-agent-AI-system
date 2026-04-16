"""LLM core services for provider routing, token accounting, and cost tracking."""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from google import genai
import tiktoken
from openai import OpenAI

from core.config import config

logger = logging.getLogger(__name__)


@dataclass
class SessionCost:
    """In-memory running totals for the current process session."""

    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0


class LLMCore:
    """Unified LLM access layer for OpenAI and Gemini models."""

    PRICING_PER_1K: dict[str, dict[str, float]] = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
        "gemini-2.0-flash": {"input": 0.00035, "output": 0.00105},
        "gemini-2.0-pro": {"input": 0.00125, "output": 0.005},
        "gemini-2.5-flash": {"input": 0.00035, "output": 0.00105},
        "gemini-2.5-pro": {"input": 0.00125, "output": 0.005},
    }

    def __init__(self) -> None:
        """Initialize LLM clients and cost logging storage."""
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None
        self.gemini_client = genai.Client(api_key=config.GEMINI_API_KEY) if config.GEMINI_API_KEY else None
        self.db_path = config.DB_PATH
        self.session_cost = SessionCost()
        self._init_db()

    def _init_db(self) -> None:
        """Create cost logging table if it does not already exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cost_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    cost_usd REAL NOT NULL,
                    latency_ms REAL NOT NULL
                )
                """
            )
            conn.commit()

    def chat(self, messages: list[dict[str, str]], model: str | None = None, temperature: float = 0.2) -> str:
        """Send a chat request to the appropriate provider with retries and logging."""
        selected_model = model or config.DEFAULT_MODEL
        if not config.OPENAI_API_KEY and not config.GEMINI_API_KEY:
            raise RuntimeError(
                "No API key loaded. Set GEMINI_API_KEY (or GOOGLE_API_KEY / GENAI_API_KEY) in .env, "
                "or set OPENAI_API_KEY if you want GPT models."
            )
        prompt_text = "\n".join(m.get("content", "") for m in messages)
        prompt_tokens = self.count_tokens(prompt_text, selected_model)

        retries = [2, 4, 8]
        last_error: Exception | None = None

        for attempt, backoff_seconds in enumerate(retries, start=1):
            start_time = time.perf_counter()
            try:
                if selected_model.startswith("gpt-"):
                    response_text, completion_tokens = self._chat_openai(messages, selected_model, temperature)
                elif selected_model.startswith("gemini-"):
                    response_text, completion_tokens = self._chat_gemini(messages, selected_model, temperature)
                else:
                    raise ValueError(f"Unsupported model prefix for model: {selected_model}")

                latency_ms = (time.perf_counter() - start_time) * 1000
                cost_usd = self._calculate_cost(selected_model, prompt_tokens, completion_tokens)
                self._log_cost(selected_model, prompt_tokens, completion_tokens, cost_usd, latency_ms)
                self._update_session_totals(prompt_tokens, completion_tokens, cost_usd, latency_ms)
                return response_text
            except (ValueError, RuntimeError, ConnectionError, TimeoutError) as exc:
                last_error = exc
                logger.warning("LLM call failed on attempt %s/%s: %s", attempt, len(retries), exc)
                if attempt < len(retries):
                    time.sleep(backoff_seconds)
            except Exception as exc:  # pylint: disable=broad-except
                last_error = exc
                logger.exception("Unexpected LLM error on attempt %s/%s", attempt, len(retries))
                if attempt < len(retries):
                    time.sleep(backoff_seconds)

        raise RuntimeError(f"LLM call failed after retries: {last_error}")

    def _chat_openai(self, messages: list[dict[str, str]], model: str, temperature: float) -> tuple[str, int]:
        """Call OpenAI chat completions API."""
        if not self.openai_client:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            content = response.choices[0].message.content or ""
            completion_tokens = response.usage.completion_tokens if response.usage else self.count_tokens(content, model)
            return content, completion_tokens
        except Exception as exc:  # pylint: disable=broad-except
            raise ConnectionError(f"OpenAI request failed: {exc}") from exc

    def _chat_gemini(self, messages: list[dict[str, str]], model: str, temperature: float) -> tuple[str, int]:
        """Call Google Gemini chat API."""
        if not self.gemini_client:
            raise RuntimeError("GEMINI_API_KEY is not configured")

        system_parts: list[str] = []
        user_parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                system_parts.append(msg.get("content", ""))
            else:
                user_parts.append(f"{role}: {msg.get('content', '')}")

        prompt_text = "\n".join(user_parts)
        system_instruction = "\n".join(system_parts).strip() or None

        try:
            response = self.gemini_client.models.generate_content(
                model=model,
                contents=prompt_text,
                config=genai.types.GenerateContentConfig(
                    temperature=temperature,
                    system_instruction=system_instruction,
                ),
            )
            content = getattr(response, "text", "") or ""
            completion_tokens = self.count_tokens(content, model)
            return content, completion_tokens
        except Exception as exc:  # pylint: disable=broad-except
            raise ConnectionError(f"Gemini request failed: {exc}") from exc

    def count_tokens(self, text: str, model: str) -> int:
        """Count approximate number of tokens for a text using model encoding."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text or ""))

    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate token cost in USD based on configured model pricing."""
        pricing_key = model
        if model not in self.PRICING_PER_1K:
            if model.startswith("gpt-4"):
                pricing_key = "gpt-4"
            elif model.startswith("gpt-3.5-turbo"):
                pricing_key = "gpt-3.5-turbo"
            elif model.startswith("gemini-2.0-flash"):
                pricing_key = "gemini-2.0-flash"
            elif model.startswith("gemini-2.0-pro"):
                pricing_key = "gemini-2.0-pro"
            elif model.startswith("gemini-2.5-flash"):
                pricing_key = "gemini-2.5-flash"
            elif model.startswith("gemini-2.5-pro"):
                pricing_key = "gemini-2.5-pro"
            else:
                pricing_key = "gpt-3.5-turbo"

        price = self.PRICING_PER_1K[pricing_key]
        return (prompt_tokens / 1000.0) * price["input"] + (completion_tokens / 1000.0) * price["output"]

    def _log_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        latency_ms: float,
    ) -> None:
        """Persist one LLM usage event to SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO cost_log (
                    timestamp,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    cost_usd,
                    latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.utcnow().isoformat(),
                    model,
                    prompt_tokens,
                    completion_tokens,
                    cost_usd,
                    latency_ms,
                ),
            )
            conn.commit()

    def _update_session_totals(
        self, prompt_tokens: int, completion_tokens: int, cost_usd: float, latency_ms: float
    ) -> None:
        """Update process-local session counters."""
        self.session_cost.total_calls += 1
        self.session_cost.total_prompt_tokens += prompt_tokens
        self.session_cost.total_completion_tokens += completion_tokens
        self.session_cost.total_cost_usd += cost_usd
        self.session_cost.total_latency_ms += latency_ms

    def get_session_cost(self) -> dict[str, float]:
        """Return aggregate session cost and latency metrics."""
        total_tokens = self.session_cost.total_prompt_tokens + self.session_cost.total_completion_tokens
        avg_latency = (
            self.session_cost.total_latency_ms / self.session_cost.total_calls
            if self.session_cost.total_calls > 0
            else 0.0
        )
        return {
            "total_calls": float(self.session_cost.total_calls),
            "total_tokens": float(total_tokens),
            "total_cost_usd": float(round(self.session_cost.total_cost_usd, 6)),
            "avg_latency_ms": float(round(avg_latency, 2)),
        }
