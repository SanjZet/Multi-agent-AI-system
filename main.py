"""CLI entrypoint for the production-grade multi-agent AI system."""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path
from typing import Iterable

from PyPDF2 import PdfReader
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from agents.critic import CriticAgent
from agents.executor import ExecutorAgent
from agents.orchestrator import OrchestratorAgent
from agents.planner import PlannerAgent
from agents.retriever import RetrieverAgent
from core.config import config
from core.llm import LLMCore
from core.vector_store import VectorStore
from memory.long_term import LongTermMemory
from memory.short_term import ShortTermMemory
from optimization.cost_optimizer import CostOptimizer
from specialized.repo_explainer import RepoExplainerAgent
from specialized.youtube_agent import YouTubeAgent
from tools.code_tool import CodeTool
from tools.repo_tool import RepoTool
from tools.web_search_tool import WebSearchTool
from tools.youtube_tool import YouTubeTool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


class DocumentLoader:
    """Loads local PDF and code documents for ingestion into the vector store."""

    CODE_EXTS = {".py", ".js", ".ts", ".go", ".java", ".md", ".txt"}

    def load_pdf(self, pdf_path: str) -> list[dict]:
        """Extract text pages from a PDF into document chunks."""
        docs: list[dict] = []
        reader = PdfReader(pdf_path)
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(
                    {
                        "text": text,
                        "source": pdf_path,
                        "metadata": {"type": "pdf", "page": idx + 1},
                    }
                )
        return docs

    def load_directory(self, directory: str) -> list[dict]:
        """Read supported code/text files from a local directory recursively."""
        docs: list[dict] = []
        root = Path(directory)
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in self.CODE_EXTS:
                continue
            try:
                with path.open("r", encoding="utf-8", errors="ignore") as fp:
                    content = fp.read()
                docs.append(
                    {
                        "text": content,
                        "source": str(path),
                        "metadata": {"type": "file", "relative_path": str(path.relative_to(root))},
                    }
                )
            except OSError:
                continue
        return docs


class Application:
    """Application container wiring all system components for CLI commands."""

    def __init__(self) -> None:
        """Initialize all components using constructor injection."""
        self.console = Console()
        self.llm = LLMCore()
        self.vector_store = VectorStore(config.VECTOR_STORE_PATH)
        self.short_term = ShortTermMemory(llm=self.llm, max_messages=20)
        self.long_term = LongTermMemory(config.DB_PATH)

        self.youtube_tool = YouTubeTool()
        self.code_tool = CodeTool()
        self.web_search_tool = WebSearchTool()
        self.repo_tool = RepoTool()

        self.retriever_agent = RetrieverAgent(self.vector_store, self.llm)
        self.planner_agent = PlannerAgent(self.llm)
        self.executor_agent = ExecutorAgent(
            tools={
                "retriever": self.retriever_agent,
                "code_executor": self.code_tool,
                "web_search": self.web_search_tool,
                "youtube_tool": self.youtube_tool,
                "repo_tool": self.repo_tool,
            },
            llm=self.llm,
        )
        self.critic_agent = CriticAgent(self.llm)
        self.orchestrator = OrchestratorAgent(
            planner=self.planner_agent,
            executor=self.executor_agent,
            critic=self.critic_agent,
            llm=self.llm,
        )

        self.youtube_agent = YouTubeAgent(self.youtube_tool, self.vector_store, self.llm)
        self.repo_explainer = RepoExplainerAgent(self.repo_tool, self.vector_store, self.llm)
        self.cost_optimizer = CostOptimizer(self.llm, config.REDIS_URL)
        self.document_loader = DocumentLoader()

    def ask(self, question: str) -> None:
        """Run orchestrator for a question and print answer, sources, and cost."""
        if not config.OPENAI_API_KEY and not config.GEMINI_API_KEY:
            self.console.print(
                Panel(
                    "No API key loaded. Create a .env file with GEMINI_API_KEY (or GOOGLE_API_KEY / GENAI_API_KEY)\n"
                    "and rerun the command.",
                    title="Missing Configuration",
                    border_style="red",
                )
            )
            return

        with self.console.status("Thinking...", spinner="dots"):
            response = self.orchestrator.run(question)

        self.short_term.add("user", question)
        self.short_term.add("assistant", response.final_answer)

        self.console.print(Panel(response.final_answer, title="Final Answer", border_style="green"))

        source_table = Table(title="Task Results")
        source_table.add_column("Task")
        source_table.add_column("Success")
        source_table.add_column("Output/Error")
        for task_id, result in response.results.items():
            content = result.get("output") or result.get("error") or ""
            source_table.add_row(str(task_id), str(result.get("success", False)), str(content)[:140])
        self.console.print(source_table)

        cost = self.llm.get_session_cost()
        cost_table = Table(title="Cost Summary")
        cost_table.add_column("Metric")
        cost_table.add_column("Value")
        cost_table.add_row("Total Calls", str(int(cost["total_calls"])))
        cost_table.add_row("Total Tokens", str(int(cost["total_tokens"])))
        cost_table.add_row("Total Cost (USD)", f"{cost['total_cost_usd']:.6f}")
        cost_table.add_row("Avg Latency (ms)", f"{cost['avg_latency_ms']:.2f}")
        self.console.print(cost_table)

    def ingest(self, path_or_url: str) -> None:
        """Ingest content from repository, YouTube, PDF, or local source directory."""
        chunks_added = 0

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=self.console) as progress:
            task = progress.add_task("Ingesting content...", total=None)

            if "github.com" in path_or_url:
                report = self.repo_explainer.explain(path_or_url)
                docs = [
                    {
                        "text": summary,
                        "source": module,
                        "metadata": {"type": "module_summary", "repo": report.url},
                    }
                    for module, summary in report.module_summaries.items()
                ]
                chunks_added = self.vector_store.add_documents(docs)
                self.console.print(Panel(report.architecture_report_md, title="Repository Architecture Report"))
            elif "youtube.com" in path_or_url or "youtu.be" in path_or_url:
                report = self.youtube_agent.process(path_or_url)
                self.console.print(Panel(report.notes_markdown, title=f"YouTube Notes: {report.title}"))
                chunks_added = report.transcript_chunks
            else:
                path = Path(path_or_url)
                if path.is_file() and path.suffix.lower() == ".pdf":
                    docs = self.document_loader.load_pdf(str(path))
                    chunks_added = self.vector_store.add_documents(docs)
                elif path.is_dir():
                    docs = self.document_loader.load_directory(str(path))
                    chunks_added = self.vector_store.add_documents(docs)
                else:
                    raise ValueError("Unsupported ingest input. Use GitHub URL, YouTube URL, PDF path, or directory path.")

            progress.update(task, description="Ingestion complete")

        self.console.print(Panel(f"Chunks added: {chunks_added}", title="Ingestion Summary", border_style="cyan"))

    def status(self) -> None:
        """Print system status including storage, cost, cache, and memory metrics."""
        docs_count = len(self.vector_store.documents)
        cost = self.llm.get_session_cost()
        cache_stats = self.cost_optimizer.get_stats()

        with sqlite3.connect(config.DB_PATH) as conn:
            mem_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

        table = Table(title="System Status")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("Total documents in vector store", str(docs_count))
        table.add_row("Session LLM calls", str(int(cost["total_calls"])))
        table.add_row("Session LLM tokens", str(int(cost["total_tokens"])))
        table.add_row("Session LLM USD", f"{cost['total_cost_usd']:.6f}")
        table.add_row("Cache hit rate", f"{cache_stats['hit_rate']:.2%}")
        table.add_row("Short-term messages", str(len(self.short_term.messages)))
        table.add_row("Long-term memories", str(mem_count))
        self.console.print(table)

    def chat(self) -> None:
        """Run interactive REPL chat loop with retained short-term memory."""
        self.console.print("Interactive chat started. Type 'exit' to quit.")
        while True:
            question = self.console.input("> ").strip()
            if question.lower() == "exit":
                self.console.print("Goodbye.")
                break
            self.ask(question)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for command-line interface."""
    parser = argparse.ArgumentParser(description="Production-grade Multi-Agent AI System")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask", help="Ask a question through the orchestrator")
    ask_parser.add_argument("question", type=str, help="Question to ask")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest data from URL or local path")
    ingest_parser.add_argument("path_or_url", type=str, help="GitHub URL, YouTube URL, PDF path, or directory")

    subparsers.add_parser("status", help="Show current system status")
    subparsers.add_parser("chat", help="Interactive chat mode")

    return parser


def main(argv: Iterable[str] | None = None) -> None:
    """Program entrypoint."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    app = Application()
    if args.command == "ask":
        app.ask(args.question)
    elif args.command == "ingest":
        app.ingest(args.path_or_url)
    elif args.command == "status":
        app.status()
    elif args.command == "chat":
        app.chat()


if __name__ == "__main__":
    main()
