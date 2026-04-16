"""LangGraph orchestrator coordinating planning, execution, and reflection loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.critic import CriticAgent, Critique
from agents.executor import ExecutorAgent
from agents.planner import Plan, PlannerAgent
from core.config import config
from core.llm import LLMCore


@dataclass
class AgentResponse:
    """Final response from orchestrator runtime."""

    final_answer: str
    plan: Optional[Plan]
    results: dict
    iterations_used: int
    critique: Optional[Critique]
    total_cost_usd: float


class AgentState(TypedDict, total=False):
    """State object passed between LangGraph nodes."""

    goal: str
    plan: Optional[Plan]
    results: dict
    current_answer: str
    critique: Optional[Critique]
    iteration: int
    final_answer: str
    status: str
    error: str


class OrchestratorAgent:
    """Coordinates all agents through a reflective execution graph."""

    def __init__(self, planner: PlannerAgent, executor: ExecutorAgent, critic: CriticAgent, llm: LLMCore) -> None:
        """Initialize orchestrator and compile LangGraph runtime."""
        self.planner = planner
        self.executor = executor
        self.critic = critic
        self.llm = llm
        self.graph = self._build_graph()

    def _build_graph(self):
        """Construct and compile the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        workflow.add_node("plan_node", self.plan_node)
        workflow.add_node("execute_node", self.execute_node)
        workflow.add_node("reflect_node", self.reflect_node)
        workflow.add_node("error_node", self.error_node)

        workflow.add_edge(START, "plan_node")
        workflow.add_edge("plan_node", "execute_node")
        workflow.add_edge("execute_node", "reflect_node")

        workflow.add_conditional_edges(
            "reflect_node",
            self._reflect_route,
            {
                "execute": "execute_node",
                "end": END,
                "error": "error_node",
            },
        )
        workflow.add_edge("error_node", END)

        return workflow.compile()

    def plan_node(self, state: AgentState) -> AgentState:
        """Generate a plan for the current goal."""
        plan = self.planner.create_plan(
            goal=state["goal"],
            available_tools=["retriever", "code_executor", "web_search", "youtube_tool", "repo_tool", "llm_only"],
            context="",
        )
        return {"plan": plan, "status": "executing"}

    def execute_node(self, state: AgentState) -> AgentState:
        """Execute current plan and synthesize an intermediate answer."""
        plan = state.get("plan")
        if not plan:
            return {"status": "failed", "error": "No plan found"}

        results = self.executor.execute_plan(plan, state.get("results", {}))
        synthesis_prompt = [
            {
                "role": "system",
                "content": "Synthesize task outputs into one coherent answer. Preserve key facts and mention uncertainties.",
            },
            {
                "role": "user",
                "content": f"Goal: {state['goal']}\n\nTask Results:\n{results}",
            },
        ]
        current_answer = self.llm.chat(synthesis_prompt)
        return {"results": results, "current_answer": current_answer, "status": "reflecting"}

    def reflect_node(self, state: AgentState) -> AgentState:
        """Evaluate current answer and decide whether to retry or finish."""
        plan = state.get("plan")
        if not plan:
            return {"status": "failed", "error": "No plan available during reflection"}

        critique = self.critic.evaluate(
            goal=state["goal"],
            answer=state.get("current_answer", ""),
            sources=[],
            plan=plan,
            results=state.get("results", {}),
        )

        iteration = int(state.get("iteration", 0))
        if critique.needs_retry and iteration < config.MAX_REFLECTION_ITERATIONS:
            revised_goal = f"{state['goal']}\n\nImprovement instruction: {critique.suggested_fix}"
            return {
                "goal": revised_goal,
                "critique": critique,
                "iteration": iteration + 1,
                "status": "executing",
            }

        return {
            "critique": critique,
            "final_answer": state.get("current_answer", ""),
            "status": "done",
        }

    def error_node(self, state: AgentState) -> AgentState:
        """Terminal error node for failed workflows."""
        return {"status": "failed", "final_answer": f"Workflow failed: {state.get('error', 'unknown error')}"}

    def _reflect_route(self, state: AgentState) -> str:
        """Route after reflection based on status and retry conditions."""
        if state.get("status") == "failed":
            return "error"
        critique = state.get("critique")
        iteration = int(state.get("iteration", 0))
        if critique and critique.needs_retry and iteration <= config.MAX_REFLECTION_ITERATIONS and state.get("status") != "done":
            return "execute"
        return "end"

    def run(self, goal: str) -> AgentResponse:
        """Execute the end-to-end orchestration graph for a user goal."""
        initial_state: AgentState = {
            "goal": goal,
            "plan": None,
            "results": {},
            "current_answer": "",
            "critique": None,
            "iteration": 0,
            "final_answer": "",
            "status": "planning",
        }

        final_state: AgentState = self.graph.invoke(initial_state)
        cost = self.llm.get_session_cost().get("total_cost_usd", 0.0)
        return AgentResponse(
            final_answer=final_state.get("final_answer", ""),
            plan=final_state.get("plan"),
            results=final_state.get("results", {}),
            iterations_used=int(final_state.get("iteration", 0)),
            critique=final_state.get("critique"),
            total_cost_usd=float(cost),
        )
