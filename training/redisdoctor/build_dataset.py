from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class TrainingExample(BaseModel):
    id: str
    source_group: str = Field(min_length=1)
    messages: list[dict[str, str]] = Field(min_length=2)


TARGETS = {"train": 600, "validation": 100, "test": 120}


def load_examples(path: Path) -> list[TrainingExample]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    examples = [TrainingExample.model_validate(item) for item in raw]
    ids = [example.id for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("样本ID必须唯一")
    return examples


def split_examples(
    examples: list[TrainingExample], seed: int = 20261008
) -> dict[str, list[TrainingExample]]:
    if len(examples) != sum(TARGETS.values()):
        raise ValueError(f"必须恰好提供{sum(TARGETS.values())}条样本")
    groups: dict[str, list[TrainingExample]] = {}
    for example in examples:
        groups.setdefault(example.source_group, []).append(example)
    shuffled_groups = list(groups.values())
    random.Random(seed).shuffle(shuffled_groups)
    splits: dict[str, list[TrainingExample]] = {name: [] for name in TARGETS}
    for group in shuffled_groups:
        candidates = sorted(
            TARGETS,
            key=lambda name: (len(splits[name]) / TARGETS[name], len(splits[name])),
        )
        placed = False
        for name in candidates:
            if len(splits[name]) + len(group) <= TARGETS[name]:
                splits[name].extend(group)
                placed = True
                break
        if not placed:
            raise ValueError("在不产生数据泄漏的前提下，来源组无法满足精确划分数量")
    if {name: len(rows) for name, rows in splits.items()} != TARGETS:
        raise ValueError("无法生成符合目标数量的数据划分")
    seen_groups: set[str] = set()
    for rows in splits.values():
        split_groups = {row.source_group for row in rows}
        if seen_groups & split_groups:
            raise ValueError("检测到source_group跨数据集泄漏")
        seen_groups |= split_groups
    return splits


def write_splits(splits: dict[str, list[TrainingExample]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = output / f"{name}.json"
        path.write_text(
            json.dumps([row.model_dump() for row in rows], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="生成无来源泄漏的RedisDoctor训练数据划分")
    parser.add_argument("source", type=Path, help="包含820条审核样本的JSON文件")
    parser.add_argument("output", type=Path, help="train/validation/test输出目录")
    args = parser.parse_args()
    write_splits(split_examples(load_examples(args.source)), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
