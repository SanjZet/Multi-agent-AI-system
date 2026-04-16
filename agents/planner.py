"""Planning agent for structured task decomposition and replanning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

from core.llm import LLMCore


@dataclass
class Task:
    """A single executable subtask in a plan."""

    task_id: str
    description: str
    required_tool: str
    depends_on: list[str]
    estimated_tokens: int
    success_criteria: str


@dataclass
class Plan:
    """A complete execution plan for a goal."""

    goal: str
    tasks: list[Task]
    created_at: datetime
    total_estimated_tokens: int


class PlannerAgent:
    """Creates and revises structured execution plans from natural language goals."""

    ALLOWED_TOOLS = {"retriever", "code_executor", "web_search", "youtube_tool", "repo_tool", "llm_only"}

    def __init__(self, llm: LLMCore) -> None:
        """Initialize planner with an LLM interface."""
        self.llm = llm

    def create_plan(self, goal: str, available_tools: list[str], context: str = "") -> Plan:
        """Create a validated plan from a goal and optional context."""
        system_prompt = (
            "You are a task decomposition engine. Given a goal, break it into the minimum ordered subtasks needed. "
            "For each subtask output JSON with fields: task_id (str), description (str), "
            "required_tool (one of: retriever, code_executor, web_search, youtube_tool, repo_tool, llm_only), "
            "depends_on (list of task_ids that must complete first), estimated_tokens (int), "
            "success_criteria (str describing what done looks like). "
            "Return a JSON array of subtasks only. No other text."
        )

        user_prompt = (
            f"Goal: {goal}\n"
            f"Available tools: {', '.join(available_tools)}\n"
            f"Context: {context or 'None'}"
        )

        raw = self.llm.chat([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
        parsed = self._parse_tasks(raw)
        self._validate_tasks(parsed)

        tasks = [
            Task(
                task_id=item["task_id"],
                description=item["description"],
                required_tool=item["required_tool"],
                depends_on=list(item.get("depends_on", [])),
                estimated_tokens=int(item.get("estimated_tokens", 0)),
                success_criteria=item.get("success_criteria", ""),
            )
            for item in parsed
        ]

        return Plan(
            goal=goal,
            tasks=tasks,
            created_at=datetime.utcnow(),
            total_estimated_tokens=sum(t.estimated_tokens for t in tasks),
        )

    def replan(self, original_plan: Plan, failed_task_id: str, error: str) -> Plan:
        """Revise an existing plan in response to a failed task."""
        serialized_plan = [task.__dict__ for task in original_plan.tasks]
        system_prompt = (
            "Revise the plan to recover from a failed task. "
            "Return only JSON array of subtasks with fields: task_id, description, required_tool, depends_on, "
            "estimated_tokens, success_criteria."
        )
        user_prompt = json.dumps(
            {
                "goal": original_plan.goal,
                "failed_task_id": failed_task_id,
                "error": error,
                "original_plan": serialized_plan,
            },
            indent=2,
        )

        raw = self.llm.chat([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
        parsed = self._parse_tasks(raw)
        self._validate_tasks(parsed)

        tasks = [
            Task(
                task_id=item["task_id"],
                description=item["description"],
                required_tool=item["required_tool"],
                depends_on=list(item.get("depends_on", [])),
                estimated_tokens=int(item.get("estimated_tokens", 0)),
                success_criteria=item.get("success_criteria", ""),
            )
            for item in parsed
        ]

        return Plan(
            goal=original_plan.goal,
            tasks=tasks,
            created_at=datetime.utcnow(),
            total_estimated_tokens=sum(t.estimated_tokens for t in tasks),
        )

    def _parse_tasks(self, raw: str) -> list[dict]:
        """Parse JSON task list from model output."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("[")
            end = raw.rfind("]")
            if start >= 0 and end > start:
                return json.loads(raw[start : end + 1])
            raise ValueError("Planner output is not valid JSON array")

    def _validate_tasks(self, tasks: list[dict]) -> None:
        """Validate task schema and dependency constraints."""
        task_ids = {item.get("task_id") for item in tasks}
        if None in task_ids or "" in task_ids:
            raise ValueError("Every task must include a non-empty task_id")

        for item in tasks:
            if item.get("required_tool") not in self.ALLOWED_TOOLS:
                raise ValueError(f"Unsupported required_tool: {item.get('required_tool')}")
            for dep in item.get("depends_on", []):
                if dep not in task_ids:
                    raise ValueError(f"Task dependency '{dep}' does not reference a valid task_id")

        graph = {item["task_id"]: list(item.get("depends_on", [])) for item in tasks}
        self._assert_acyclic(graph)

    def _assert_acyclic(self, graph: dict[str, list[str]]) -> None:
        """Raise ValueError if dependency graph contains a cycle."""
        visited: set[str] = set()
        visiting: set[str] = set()

        def dfs(node: str) -> None:
            if node in visiting:
                raise ValueError("Circular dependency detected in plan")
            if node in visited:
                return
            visiting.add(node)
            for dep in graph.get(node, []):
                dfs(dep)
            visiting.remove(node)
            visited.add(node)

        for node in graph:
            dfs(node)
