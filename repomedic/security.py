from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


class UnsafeOperation(ValueError):
    """仓库工具请求越过安全边界时抛出。"""


class CommandPolicy:
    DEFAULT_PREFIXES: tuple[tuple[str, ...], ...] = (
        ("python", "-m", "pytest"),
        ("python", "-m", "unittest"),
        ("pytest",),
        ("git", "diff"),
        ("git", "status"),
        ("ruff", "check"),
        ("mypy",),
    )
    _SHELL_META = re.compile(r"[;&|><`\r\n]")

    def __init__(self, allowed_prefixes: Iterable[tuple[str, ...]] | None = None) -> None:
        self.allowed_prefixes = tuple(allowed_prefixes or self.DEFAULT_PREFIXES)

    def validate(self, command: list[str]) -> tuple[str, ...]:
        if not command or any(not isinstance(part, str) or not part for part in command):
            raise UnsafeOperation("命令必须由非空字符串参数组成")
        if any(self._SHELL_META.search(part) for part in command):
            raise UnsafeOperation("不允许使用Shell元字符")
        normalized = tuple(command)
        if not any(normalized[: len(prefix)] == prefix for prefix in self.allowed_prefixes):
            raise UnsafeOperation(f"命令不在白名单中：{command[0]}")
        return normalized


def safe_repository_path(root: Path, relative_path: str | Path) -> Path:
    root = root.resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise UnsafeOperation("路径超出了仓库根目录") from exc
    return candidate
