"""Repository utilities for cloning, inspection, and lightweight static analysis."""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

from git import Repo

logger = logging.getLogger(__name__)


class RepoTool:
    """Tools for reading and analyzing source repositories."""

    IGNORE_DIRS = {".git", "node_modules", "__pycache__", "dist", "build"}
    IGNORE_FILES = {".env"}
    ENTRY_CANDIDATES = ["main.py", "app.py", "index.py", "server.py", "manage.py"]

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

    def clone(self, url: str, dest: str = "/tmp/repo") -> str:
        """Clone a git repository into the destination directory."""
        start = time.perf_counter()
        payload = {"url": url, "dest": dest}
        try:
            dest_path = Path(dest)
            if dest_path.exists():
                shutil.rmtree(dest_path)
            Repo.clone_from(url, str(dest_path))
            latency = (time.perf_counter() - start) * 1000
            self._log_tool_call("repo_clone", payload, True, latency)
            return str(dest_path)
        except (OSError, RuntimeError, ValueError) as exc:
            latency = (time.perf_counter() - start) * 1000
            self._log_tool_call("repo_clone", payload, False, latency)
            logger.exception("Repo clone failed")
            raise RuntimeError(f"Failed to clone repository: {exc}") from exc

    def get_file_tree(self, repo_path: str) -> str:
        """Return an indented file tree string while ignoring noisy paths."""
        start = time.perf_counter()
        payload = {"repo_path": repo_path}
        try:
            root = Path(repo_path)
            lines: list[str] = []
            for current_root, dirs, files in os.walk(root):
                dirs[:] = [d for d in dirs if d not in self.IGNORE_DIRS]
                rel_root = Path(current_root).relative_to(root)
                depth = len(rel_root.parts)
                indent = "  " * depth
                dir_name = root.name if str(rel_root) == "." else rel_root.name
                lines.append(f"{indent}{dir_name}/")

                for file_name in sorted(files):
                    if file_name in self.IGNORE_FILES or fnmatch.fnmatch(file_name, "*.pyc"):
                        continue
                    lines.append(f"{indent}  {file_name}")

            tree = "\n".join(lines)
            latency = (time.perf_counter() - start) * 1000
            self._log_tool_call("repo_get_file_tree", payload, True, latency)
            return tree
        except (OSError, ValueError) as exc:
            latency = (time.perf_counter() - start) * 1000
            self._log_tool_call("repo_get_file_tree", payload, False, latency)
            logger.exception("Tree generation failed")
            raise RuntimeError(f"Failed to build file tree: {exc}") from exc

    def read_source_files(self, repo_path: str) -> list[dict]:
        """Read supported source files and return content with metadata."""
        start = time.perf_counter()
        payload = {"repo_path": repo_path}
        extensions = {".py": "python", ".js": "javascript", ".ts": "typescript", ".go": "go", ".java": "java", ".md": "markdown"}
        sources: list[dict] = []

        try:
            root = Path(repo_path)
            for current_root, dirs, files in os.walk(root):
                dirs[:] = [d for d in dirs if d not in self.IGNORE_DIRS]
                for file_name in files:
                    file_path = Path(current_root) / file_name
                    if file_name in self.IGNORE_FILES or fnmatch.fnmatch(file_name, "*.pyc"):
                        continue
                    ext = file_path.suffix.lower()
                    if ext not in extensions:
                        continue
                    try:
                        with file_path.open("r", encoding="utf-8", errors="ignore") as fp:
                            content = fp.read()
                        sources.append(
                            {
                                "path": str(file_path.relative_to(root)),
                                "content": content,
                                "language": extensions[ext],
                                "size_lines": len(content.splitlines()),
                            }
                        )
                    except OSError:
                        logger.warning("Could not read file: %s", file_path)

            latency = (time.perf_counter() - start) * 1000
            self._log_tool_call("repo_read_source_files", payload, True, latency)
            return sources
        except (OSError, ValueError) as exc:
            latency = (time.perf_counter() - start) * 1000
            self._log_tool_call("repo_read_source_files", payload, False, latency)
            logger.exception("Reading source files failed")
            raise RuntimeError(f"Failed to read source files: {exc}") from exc

    def build_dependency_graph(self, repo_path: str) -> dict:
        """Build a Python import dependency graph from AST parsing."""
        start = time.perf_counter()
        payload = {"repo_path": repo_path}
        graph: dict[str, list[str]] = {}

        try:
            root = Path(repo_path)
            python_files = list(root.rglob("*.py"))
            for py_file in python_files:
                if any(part in self.IGNORE_DIRS for part in py_file.parts):
                    continue
                try:
                    with py_file.open("r", encoding="utf-8", errors="ignore") as fp:
                        content = fp.read()
                    tree = ast.parse(content)
                except (OSError, SyntaxError):
                    continue

                imports: list[str] = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports.extend(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom):
                        module_name = node.module or ""
                        imports.append(module_name)
                graph[str(py_file.relative_to(root))] = sorted(set(imports))

            latency = (time.perf_counter() - start) * 1000
            self._log_tool_call("repo_dependency_graph", payload, True, latency)
            return graph
        except (OSError, ValueError) as exc:
            latency = (time.perf_counter() - start) * 1000
            self._log_tool_call("repo_dependency_graph", payload, False, latency)
            logger.exception("Dependency graph failed")
            raise RuntimeError(f"Failed to build dependency graph: {exc}") from exc

    def find_entry_point(self, repo_path: str) -> str:
        """Heuristically identify a likely repository entry point."""
        start = time.perf_counter()
        payload = {"repo_path": repo_path}
        try:
            root = Path(repo_path)
            for candidate in self.ENTRY_CANDIDATES:
                match = list(root.rglob(candidate))
                for path in match:
                    if any(part in self.IGNORE_DIRS for part in path.parts):
                        continue
                    rel = str(path.relative_to(root))
                    latency = (time.perf_counter() - start) * 1000
                    self._log_tool_call("repo_find_entry_point", payload, True, latency)
                    return rel

            latency = (time.perf_counter() - start) * 1000
            self._log_tool_call("repo_find_entry_point", payload, True, latency)
            return "unknown"
        except (OSError, ValueError) as exc:
            latency = (time.perf_counter() - start) * 1000
            self._log_tool_call("repo_find_entry_point", payload, False, latency)
            logger.exception("Entry point detection failed")
            raise RuntimeError(f"Failed to find entry point: {exc}") from exc
