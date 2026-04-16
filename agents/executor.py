"""Execution agent that routes plan tasks to tools or direct LLM calls."""

from __future__ import annotations

import time
from dataclasses import dataclass

from agents.planner import Plan, Task
from core.llm import LLMCore


@dataclass
class TaskResult:
    """Result of executing a single plan task."""

    task_id: str
    output: str
    error: str
    success: bool
    execution_time_ms: float
    tokens_used: int


class ExecutorAgent:
    """Executes tasks by selecting the right backend tool."""

    def __init__(self, tools: dict, llm: LLMCore) -> None:
        """Initialize with a tool registry and LLM core."""
        self.tools = tools
        self.llm = llm

    def execute_task(self, task: Task, context: dict) -> TaskResult:
        """Execute a single task and return structured result metadata."""
        start = time.perf_counter()
        tokens_used = 0
        try:
            if task.required_tool == "llm_only":
                context_blob = str(context)
                prompt = [
                    {"role": "system", "content": "Solve the task directly and concisely."},
                    {"role": "user", "content": f"Task: {task.description}\n\nContext: {context_blob}"},
                ]
                output = self.llm.chat(prompt)
                tokens_used = self.llm.count_tokens(prompt[0]["content"] + prompt[1]["content"] + output, "gpt-3.5-turbo")
            elif task.required_tool == "retriever":
                tool = self.tools["retriever"]
                query = context.get("query", task.description)
                result = tool.retrieve(query)
                output = result.answer
                tokens_used = result.context_tokens_used
            elif task.required_tool == "code_executor":
                tool = self.tools["code_executor"]
                code = context.get("code", task.description)
                result = tool.execute_python(code)
                output = result.get("output", "")
                if result.get("error"):
                    output = f"{output}\nERROR: {result['error']}".strip()
            elif task.required_tool == "web_search":
                tool = self.tools["web_search"]
                query = context.get("query", task.description)
                search_results = tool.search(query)
                output = tool.summarize_results(search_results, query, self.llm)
            elif task.required_tool == "youtube_tool":
                tool = self.tools["youtube_tool"]
                url = context.get("url", task.description)
                output = str(tool.get_metadata(url))
            elif task.required_tool == "repo_tool":
                tool = self.tools["repo_tool"]
                if "github.com" in task.description and not context.get("repo_path"):
                    repo_path = tool.clone(task.description)
                else:
                    repo_path = context.get("repo_path", ".")
                output = tool.get_file_tree(repo_path)
            else:
                raise ValueError(f"Unknown required_tool: {task.required_tool}")

            latency = (time.perf_counter() - start) * 1000
            return TaskResult(
                task_id=task.task_id,
                output=output,
                error="",
                success=True,
                execution_time_ms=float(round(latency, 2)),
                tokens_used=tokens_used,
            )
        except Exception as exc:  # pylint: disable=broad-except
            latency = (time.perf_counter() - start) * 1000
            return TaskResult(
                task_id=task.task_id,
                output="",
                error=str(exc),
                success=False,
                execution_time_ms=float(round(latency, 2)),
                tokens_used=tokens_used,
            )

    def execute_plan(self, plan: Plan, results_so_far: dict) -> dict:
        """Execute all tasks whose dependencies are met, until progress stops."""
        results = dict(results_so_far)
        pending = {task.task_id: task for task in plan.tasks if task.task_id not in results}

        while pending:
            progress = False
            for task_id, task in list(pending.items()):
                if all(dep in results and results[dep]["success"] for dep in task.depends_on):
                    context = {
                        "results_so_far": results,
                        "query": plan.goal,
                    }
                    result = self.execute_task(task, context)
                    results[task_id] = {
                        "success": result.success,
                        "output": result.output,
                        "error": result.error,
                        "execution_time_ms": result.execution_time_ms,
                        "tokens_used": result.tokens_used,
                    }
                    pending.pop(task_id)
                    progress = True

            if not progress:
                for task_id, task in pending.items():
                    unmet = [dep for dep in task.depends_on if dep not in results]
                    results[task_id] = {
                        "success": False,
                        "output": "",
                        "error": f"Unresolved dependencies: {unmet}",
                        "execution_time_ms": 0.0,
                        "tokens_used": 0,
                    }
                break

        return results
