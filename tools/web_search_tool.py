"""Web search and page extraction utilities with SerpAPI and DuckDuckGo fallback."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup

from core.config import config
from core.llm import LLMCore

logger = logging.getLogger(__name__)


class WebSearchTool:
    """Searches web results and synthesizes findings."""

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

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Search results using SerpAPI first, then DuckDuckGo HTML fallback."""
        start = time.perf_counter()
        payload = {"query": query, "k": k}

        try:
            if config.SERPAPI_KEY:
                serp_results = self._search_serpapi(query, k)
                if serp_results:
                    latency = (time.perf_counter() - start) * 1000
                    self._log_tool_call("web_search", payload, True, latency)
                    return serp_results
        except requests.RequestException:
            logger.warning("SerpAPI search failed, falling back to DuckDuckGo")

        try:
            ddg_results = self._search_duckduckgo(query, k)
            latency = (time.perf_counter() - start) * 1000
            self._log_tool_call("web_search", payload, True, latency)
            return ddg_results
        except requests.RequestException as exc:
            latency = (time.perf_counter() - start) * 1000
            self._log_tool_call("web_search", payload, False, latency)
            logger.exception("Web search failed")
            raise RuntimeError(f"Search failed: {exc}") from exc

    def _search_serpapi(self, query: str, k: int) -> list[dict]:
        """Run Google search via SerpAPI."""
        params = {
            "engine": "google",
            "q": query,
            "api_key": config.SERPAPI_KEY,
            "num": k,
        }
        resp = requests.get("https://serpapi.com/search", params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        results: list[dict] = []
        for item in data.get("organic_results", [])[:k]:
            results.append(
                {
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "url": item.get("link", ""),
                    "source": "serpapi",
                }
            )
        return results

    def _search_duckduckgo(self, query: str, k: int) -> list[dict]:
        """Run a DuckDuckGo HTML results scrape."""
        params = urlencode({"q": query})
        url = f"https://duckduckgo.com/html/?{params}"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        results: list[dict] = []

        for result in soup.select("div.result")[:k]:
            title_tag = result.select_one("a.result__a")
            snippet_tag = result.select_one("a.result__snippet") or result.select_one("div.result__snippet")
            if not title_tag:
                continue
            results.append(
                {
                    "title": title_tag.get_text(strip=True),
                    "snippet": snippet_tag.get_text(" ", strip=True) if snippet_tag else "",
                    "url": title_tag.get("href", ""),
                    "source": "duckduckgo",
                }
            )
        return results

    def fetch_page(self, url: str, max_chars: int = 3000) -> str:
        """Fetch and clean page text by removing navigation and script noise."""
        start = time.perf_counter()
        payload = {"url": url, "max_chars": max_chars}

        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            for tag_name in ["script", "style", "nav", "footer", "header", "aside"]:
                for tag in soup.find_all(tag_name):
                    tag.decompose()

            text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
            cleaned = text[:max_chars]
            latency = (time.perf_counter() - start) * 1000
            self._log_tool_call("web_fetch_page", payload, True, latency)
            return cleaned
        except requests.RequestException as exc:
            latency = (time.perf_counter() - start) * 1000
            self._log_tool_call("web_fetch_page", payload, False, latency)
            logger.exception("Page fetch failed")
            raise RuntimeError(f"Failed to fetch page: {exc}") from exc

    def summarize_results(self, results: list[dict], query: str, llm: LLMCore) -> str:
        """Summarize result snippets into two paragraphs with source citations."""
        start = time.perf_counter()
        payload = {"query": query, "results_count": len(results)}

        try:
            lines = []
            for idx, item in enumerate(results, start=1):
                lines.append(
                    f"[{idx}] {item.get('title', '')}\nURL: {item.get('url', '')}\nSnippet: {item.get('snippet', '')}"
                )

            messages = [
                {
                    "role": "system",
                    "content": (
                        "Synthesize the search results into exactly two short paragraphs. "
                        "Cite supporting result ids inline like [1], [2]."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nResults:\n" + "\n\n".join(lines),
                },
            ]
            summary = llm.chat(messages)
            latency = (time.perf_counter() - start) * 1000
            self._log_tool_call("web_summarize_results", payload, True, latency)
            return summary
        except (RuntimeError, ValueError) as exc:
            latency = (time.perf_counter() - start) * 1000
            self._log_tool_call("web_summarize_results", payload, False, latency)
            logger.exception("Search summary failed")
            raise RuntimeError(f"Failed to summarize results: {exc}") from exc
