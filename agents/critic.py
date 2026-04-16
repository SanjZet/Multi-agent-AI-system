"""Critic agent for quality evaluation and retry decisions."""

from __future__ import annotations

import json
from dataclasses import dataclass

from agents.planner import Plan
from core.llm import LLMCore


@dataclass
class Critique:
    """Structured evaluation returned by the critic."""

    scores: dict
    overall_score: float
    weaknesses: list[str]
    suggested_fix: str
    needs_retry: bool


class CriticAgent:
    """Evaluates answer quality and suggests concrete improvements."""

    def __init__(self, llm: LLMCore) -> None:
        """Initialize critic with LLM backend."""
        self.llm = llm

    def evaluate(self, goal: str, answer: str, sources: list[dict], plan: Plan, results: dict) -> Critique:
        """Evaluate the answer against quality criteria and compute overall score."""
        prompt = (
            "You are a rigorous answer quality critic. "
            "Evaluate the answer on these dimensions: "
            "1. factual_accuracy (0-10): are all claims grounded? "
            "2. completeness (0-10): does it fully address the goal? "
            "3. coherence (0-10): is it logically structured? "
            "4. hallucination_risk (0-10, 0=none, 10=severe): are any claims not backed by sources? "
            "5. weaknesses: list of specific problems found. "
            "6. suggested_fix: one concrete instruction to improve. "
            "Return valid JSON only matching this exact schema."
        )

        user_payload = {
            "goal": goal,
            "answer": answer,
            "sources": sources,
            "plan": [t.__dict__ for t in plan.tasks],
            "results": results,
        }
        raw = self.llm.chat([{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps(user_payload)}])
        data = self._parse_json(raw)

        accuracy = float(data.get("factual_accuracy", 0))
        completeness = float(data.get("completeness", 0))
        coherence = float(data.get("coherence", 0))
        hallucination = float(data.get("hallucination_risk", 10))
        overall_score = (accuracy * 0.4 + completeness * 0.3 + coherence * 0.2 + (10 - hallucination) * 0.1) / 10

        return Critique(
            scores={
                "factual_accuracy": accuracy,
                "completeness": completeness,
                "coherence": coherence,
                "hallucination_risk": hallucination,
            },
            overall_score=overall_score,
            weaknesses=[str(item) for item in data.get("weaknesses", [])],
            suggested_fix=str(data.get("suggested_fix", "Improve evidence grounding and completeness.")),
            needs_retry=overall_score < 0.8,
        )

    def _parse_json(self, raw: str) -> dict:
        """Parse JSON object from model output safely."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                return json.loads(raw[start : end + 1])
            raise ValueError("Critic output was not valid JSON")
