"""Cost optimization utilities for model routing, caching, and batch execution."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any

import redis

from core.config import config
from core.llm import LLMCore

logger = logging.getLogger(__name__)


class CostOptimizer:
    """Optimizes LLM spend with caching, model routing, and async batching."""

    def __init__(self, llm: LLMCore, redis_url: str) -> None:
        """Initialize cache backend and runtime statistics."""
        self.llm = llm
        self.cache_hits = 0
        self.cache_misses = 0
        self.estimated_savings_usd = 0.0
        self._fallback_cache: dict[str, str] = {}
        self.redis_client = None

        try:
            self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
        except redis.RedisError:
            logger.warning("Redis unavailable. Falling back to in-process cache.")
            self.redis_client = None

    def cached_chat(self, messages: list[dict], model: str, ttl: int = 3600) -> str:
        """Return cached chat response when available, otherwise call model and cache result."""
        key_payload = model + json.dumps(messages, sort_keys=True, default=str)
        cache_key = hashlib.sha256(key_payload.encode("utf-8")).hexdigest()

        cached = self._cache_get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        self.cache_misses += 1
        response = self.llm.chat(messages, model=model)
        self._cache_set(cache_key, response, ttl)

        prompt_tokens = self.llm.count_tokens("\n".join(m.get("content", "") for m in messages), model)
        output_tokens = self.llm.count_tokens(response, model)
        estimated_cost = self.llm._calculate_cost(model, prompt_tokens, output_tokens)  # pylint: disable=protected-access
        self.estimated_savings_usd += estimated_cost

        return response

    def _cache_get(self, cache_key: str) -> str | None:
        """Read cache value from Redis or fallback store."""
        if self.redis_client:
            try:
                value = self.redis_client.get(cache_key)
                return value if value is not None else None
            except redis.RedisError:
                return self._fallback_cache.get(cache_key)
        return self._fallback_cache.get(cache_key)

    def _cache_set(self, cache_key: str, value: str, ttl: int) -> None:
        """Store cache value in Redis or fallback store."""
        if self.redis_client:
            try:
                self.redis_client.setex(cache_key, ttl, value)
                return
            except redis.RedisError:
                self._fallback_cache[cache_key] = value
                return
        self._fallback_cache[cache_key] = value

    def route_model(self, task_description: str) -> str:
        """Classify complexity and select either fast or default model."""
        prompt = [
            {
                "role": "system",
                "content": (
                    "Is this task: (A) simple factual retrieval or formatting, or (B) complex reasoning, planning, "
                    "or multi-step analysis? Reply A or B only."
                ),
            },
            {"role": "user", "content": task_description},
        ]
        reply = self.llm.chat(prompt, model=config.FAST_MODEL, temperature=0.0).strip().upper()
        return config.FAST_MODEL if "A" in reply[:2] else config.DEFAULT_MODEL

    async def batch_calls(self, prompts: list[str], system: str) -> list[str]:
        """Run up to five concurrent LLM requests and preserve input ordering."""
        semaphore = asyncio.Semaphore(5)

        async def worker(prompt: str) -> str:
            async with semaphore:
                messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
                return await asyncio.to_thread(self.llm.chat, messages, config.FAST_MODEL, 0.2)

        tasks = [worker(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)

    def get_stats(self) -> dict[str, Any]:
        """Return cache effectiveness and estimated cost savings."""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total) if total > 0 else 0.0
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": round(hit_rate, 4),
            "estimated_savings_usd": round(self.estimated_savings_usd, 6),
        }
