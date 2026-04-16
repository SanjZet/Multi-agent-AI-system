"""Specialized agent for repository ingestion and architecture explanation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from core.llm import LLMCore
from core.vector_store import VectorStore
from tools.repo_tool import RepoTool


@dataclass
class RepoReport:
    """Structured report returned by repository explainer."""

    url: str
    file_tree: str
    entry_point: str
    dependency_graph: dict
    module_summaries: dict
    architecture_report_md: str
    total_files: int
    total_lines: int


class RepoExplainerAgent:
    """Clones a repository, summarizes modules, and generates architecture documentation."""

    def __init__(self, repo_tool: RepoTool, vector_store: VectorStore, llm: LLMCore) -> None:
        """Initialize repository explainer dependencies."""
        self.repo_tool = repo_tool
        self.vector_store = vector_store
        self.llm = llm

    def explain(self, github_url: str) -> RepoReport:
        """Generate a complete architecture report for a GitHub repository."""
        repo_hash = hashlib.sha256(github_url.encode("utf-8")).hexdigest()[:10]
        local_path = self.repo_tool.clone(github_url, dest=f"/tmp/repo_{repo_hash}")

        file_tree = self.repo_tool.get_file_tree(local_path)
        entry_point = self.repo_tool.find_entry_point(local_path)
        dependency_graph = self.repo_tool.build_dependency_graph(local_path)
        source_files = self.repo_tool.read_source_files(local_path)

        module_summaries: dict[str, str] = {}
        docs_to_add = []
        total_lines = 0

        for item in source_files:
            total_lines += int(item.get("size_lines", 0))
            prompt = [
                {
                    "role": "system",
                    "content": (
                        "In exactly 2 sentences: what does this module do and what are its key dependencies?"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Path: {item['path']}\n\nCode:\n{item['content'][:12000]}",
                },
            ]
            summary = self.llm.chat(prompt)
            module_summaries[item["path"]] = summary
            docs_to_add.append(
                {
                    "text": item["content"],
                    "source": item["path"],
                    "metadata": {
                        "type": "repository_file",
                        "repo": github_url,
                        "language": item.get("language", "unknown"),
                    },
                }
            )

        self.vector_store.add_documents(docs_to_add)

        architecture_prompt = [
            {
                "role": "system",
                "content": (
                    "Produce Architecture Report markdown with sections: "
                    "High-Level Purpose, Module Map (table: module | role | depends_on), "
                    "Data Flow (numbered), Key Design Patterns used, Entry Point walkthrough, "
                    "How to Extend (3 concrete suggestions)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Repository URL: {github_url}\n"
                    f"Entry point: {entry_point}\n"
                    f"Dependency graph: {dependency_graph}\n"
                    f"Module summaries: {module_summaries}"
                ),
            },
        ]
        architecture_report_md = self.llm.chat(architecture_prompt)

        return RepoReport(
            url=github_url,
            file_tree=file_tree,
            entry_point=entry_point,
            dependency_graph=dependency_graph,
            module_summaries=module_summaries,
            architecture_report_md=architecture_report_md,
            total_files=len(source_files),
            total_lines=total_lines,
        )
