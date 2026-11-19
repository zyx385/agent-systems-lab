from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from .models import Evidence
from .security import CommandPolicy, UnsafeOperation, safe_repository_path


class CommandResult(BaseModel):
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


class CallGraph(BaseModel):
    path: str
    definitions: dict[str, list[str]] = Field(default_factory=dict)


class RepositoryTools:
    def __init__(self, root: Path, policy: CommandPolicy | None = None, timeout: int = 30) -> None:
        self.root = root.resolve()
        if not self.root.is_dir():
            raise ValueError("仓库根目录不存在")
        self.policy = policy or CommandPolicy()
        self.timeout = timeout
        self.call_count = 0

    def read(self, relative_path: str, start: int = 1, end: int = 200) -> str:
        self.call_count += 1
        path = safe_repository_path(self.root, relative_path)
        if not path.is_file():
            raise FileNotFoundError(relative_path)
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        start_index = max(start - 1, 0)
        return "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines[start_index:end], start_index))

    def search(self, pattern: str, limit: int = 30) -> list[Evidence]:
        self.call_count += 1
        if not pattern.strip():
            return []
        expression = re.compile(re.escape(pattern), re.IGNORECASE)
        results: list[Evidence] = []
        ignored = {".git", ".venv", "venv", "node_modules", "__pycache__"}
        for path in sorted(self.root.rglob("*.py")):
            if ignored.intersection(path.parts):
                continue
            relative = path.relative_to(self.root).as_posix()
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
            ):
                if expression.search(line):
                    results.append(
                        Evidence(
                            id=f"src-{len(results) + 1}",
                            kind="source",
                            path=relative,
                            line=line_number,
                            excerpt=line.strip()[:500],
                        )
                    )
                    if len(results) >= limit:
                        return results
        return results

    def call_graph(self, relative_path: str) -> CallGraph:
        self.call_count += 1
        path = safe_repository_path(self.root, relative_path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        definitions: dict[str, list[str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                calls: list[str] = []
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name):
                            calls.append(child.func.id)
                        elif isinstance(child.func, ast.Attribute):
                            calls.append(child.func.attr)
                definitions[node.name] = sorted(set(calls))
        return CallGraph(path=relative_path, definitions=definitions)

    def run(self, command: list[str]) -> CommandResult:
        self.call_count += 1
        validated = self.policy.validate(command)
        executable_command = list(validated)
        if executable_command[0] == "python":
            executable_command[0] = sys.executable
        try:
            completed = subprocess.run(
                executable_command,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                shell=False,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                command=list(validated),
                returncode=124,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "命令执行超时",
                timed_out=True,
            )
        return CommandResult(
            command=list(validated),
            returncode=completed.returncode,
            stdout=completed.stdout[-10_000:],
            stderr=completed.stderr[-10_000:],
        )

    def apply_text(self, relative_path: str, content: str, *, approved: bool) -> None:
        if not approved:
            raise UnsafeOperation("写入仓库前必须获得明确确认")
        path = safe_repository_path(self.root, relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
