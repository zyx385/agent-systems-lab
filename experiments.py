from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field


class RunManifest(BaseModel):
    project: str
    system: str
    model: str
    dataset_version: str
    code_commit: str
    upstream_commits: dict[str, str]
    command: list[str]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    python: str = Field(default_factory=lambda: sys.version)
    platform: str = Field(default_factory=platform.platform)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")


def current_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unversioned"
