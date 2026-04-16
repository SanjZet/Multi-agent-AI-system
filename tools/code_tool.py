"""Code execution and static inspection utilities."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import logging
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime

from pyflakes.api import check
from pyflakes.reporter import Reporter

logger = logging.getLogger(__name__)


class CodeTool:
    """Provides safe-ish execution, formatting, extraction, and linting for code."""

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

    def execute_python(self, code: str, timeout: int = 10) -> dict:
        """Execute Python code in a restricted namespace and capture output."""
        start = time.perf_counter()
        payload = {"code_preview": code[:500], "timeout": timeout}

        out_buffer = io.StringIO()
        err_buffer = io.StringIO()
        finished = threading.Event()
        timeout_hit = threading.Event()
        execution_error: list[str] = []

        safe_builtins = {
            "print": print,
            "len": len,
            "range": range,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "enumerate": enumerate,
            "zip": zip,
            "any": any,
            "all": all,
        }
        globals_dict = {"__builtins__": safe_builtins}
        locals_dict: dict = {}

        def runner() -> None:
            try:
                with redirect_stdout(out_buffer), redirect_stderr(err_buffer):
                    exec(code, globals_dict, locals_dict)  # noqa: S102
            except Exception as exc:  # pylint: disable=broad-except
                execution_error.append(str(exc))
            finally:
                finished.set()

        thread = threading.Thread(target=runner, daemon=True)
        timer = threading.Timer(timeout, lambda: timeout_hit.set())

        thread.start()
        timer.start()

        while not finished.is_set() and not timeout_hit.is_set():
            time.sleep(0.01)

        success = finished.is_set() and not timeout_hit.is_set() and not execution_error
        if timeout_hit.is_set() and not finished.is_set():
            execution_error.append(f"Execution timed out after {timeout} seconds")

        timer.cancel()
        latency = (time.perf_counter() - start) * 1000
        self._log_tool_call("code_execute_python", payload, success, latency)

        return {
            "output": out_buffer.getvalue(),
            "error": "\n".join(err_buffer.getvalue().splitlines() + execution_error).strip(),
            "success": success,
            "execution_time_ms": float(round(latency, 2)),
        }

    def format_code(self, code: str, language: str) -> str:
        """Format code for supported languages, with Python Black support."""
        start = time.perf_counter()
        payload = {"language": language, "code_preview": code[:300]}
        formatted = code
        success = True

        try:
            if language.lower() == "python":
                try:
                    import black

                    formatted = black.format_str(code, mode=black.Mode())
                except ImportError:
                    formatted = code
        except Exception:  # pylint: disable=broad-except
            success = False
            logger.exception("Code formatting failed")

        latency = (time.perf_counter() - start) * 1000
        self._log_tool_call("code_format_code", payload, success, latency)
        return formatted

    def extract_functions(self, code: str) -> list[dict]:
        """Extract top-level and nested Python function definitions from source."""
        start = time.perf_counter()
        payload = {"code_preview": code[:400]}

        try:
            tree = ast.parse(code)
            functions: list[dict] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    args = [arg.arg for arg in node.args.args]
                    functions.append(
                        {
                            "name": node.name,
                            "args": args,
                            "docstring": ast.get_docstring(node) or "",
                            "start_line": int(node.lineno),
                        }
                    )
            latency = (time.perf_counter() - start) * 1000
            self._log_tool_call("code_extract_functions", payload, True, latency)
            return functions
        except SyntaxError:
            latency = (time.perf_counter() - start) * 1000
            self._log_tool_call("code_extract_functions", payload, False, latency)
            return []

    def lint_python(self, code: str) -> list[dict]:
        """Run Pyflakes lint checks and return warning/error diagnostics."""
        start = time.perf_counter()
        payload = {"code_preview": code[:400]}

        out = io.StringIO()
        err = io.StringIO()
        reporter = Reporter(out, err)
        diagnostics: list[dict] = []

        try:
            check(code, filename="<string>", reporter=reporter)
            lines = out.getvalue().splitlines() + err.getvalue().splitlines()
            for line in lines:
                parts = line.split(":", 3)
                if len(parts) >= 4:
                    _, line_no, col_no, message = parts
                    diagnostics.append(
                        {
                            "line": int(line_no.strip() or 0),
                            "col": int(col_no.strip() or 0),
                            "message": message.strip(),
                        }
                    )
                else:
                    diagnostics.append({"line": 0, "col": 0, "message": line.strip()})

            latency = (time.perf_counter() - start) * 1000
            self._log_tool_call("code_lint_python", payload, True, latency)
            return diagnostics
        except Exception:  # pylint: disable=broad-except
            latency = (time.perf_counter() - start) * 1000
            self._log_tool_call("code_lint_python", payload, False, latency)
            logger.exception("Pyflakes linting failed")
            return []
