from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from .models import RedisDocument


_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def parse_redis_markdown(
    text: str,
    *,
    source_url: str,
    source_prefix: str,
    redis_version: str | None = None,
    module: str = "general",
) -> list[RedisDocument]:
    """按Markdown标题切分文本，同时保留来源和Redis元数据。"""
    sections: list[tuple[str, list[str]]] = []
    current_title = "概述"
    current_lines: list[str] = []
    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            if any(item.strip() for item in current_lines):
                sections.append((current_title, current_lines))
            current_title = match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)
    if any(item.strip() for item in current_lines):
        sections.append((current_title, current_lines))

    documents: list[RedisDocument] = []
    for index, (title, lines) in enumerate(sections, start=1):
        body = "\n".join(lines).strip()
        if not body:
            continue
        documents.append(
            RedisDocument(
                source_id=f"{source_prefix}#{index}",
                title=title,
                text=body,
                source_url=source_url,
                redis_version=redis_version,
                module=module,
                kind="documentation",
            )
        )
    return documents


def parse_github_issues(
    records: Iterable[dict[str, Any]], *, repository_url: str, redis_version: str | None = None
) -> list[RedisDocument]:
    documents: list[RedisDocument] = []
    for record in records:
        number = record.get("number")
        title = str(record.get("title") or "").strip()
        body = str(record.get("body") or "").strip()
        if not number or not title or not body:
            continue
        labels = [
            str(item.get("name", "")) if isinstance(item, dict) else str(item)
            for item in record.get("labels", [])
        ]
        module = next((label for label in labels if label), "general")
        documents.append(
            RedisDocument(
                source_id=f"issue-{number}",
                title=title,
                text=body,
                source_url=str(record.get("html_url") or f"{repository_url}/issues/{number}"),
                redis_version=redis_version,
                module=module,
                kind="issue",
            )
        )
    return documents


def load_issue_export(path: Path, repository_url: str) -> list[RedisDocument]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("GitHub Issue导出文件必须是JSON数组")
    return parse_github_issues(raw, repository_url=repository_url)
