from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ReviewDecision(BaseModel):
    example_id: str = Field(min_length=1)
    reviewer_id: str = Field(min_length=1)
    answer_correct: bool
    citation_entails: bool
    abstention_correct: bool
    version_error: bool = False
    hallucination: bool = False
    notes: str = ""
    reviewed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("notes")
    @classmethod
    def evidence_for_failed_entailment(cls, value: str, info: Any) -> str:
        if info.data.get("citation_entails") is False and not value.strip():
            raise ValueError("引用不支持时必须填写说明")
        return value


def load_review_ledger(path: Path, known_example_ids: set[str]) -> list[ReviewDecision]:
    if not path.exists():
        return []
    rows: list[ReviewDecision] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            decision = ReviewDecision.model_validate_json(line)
        except ValueError as exc:
            raise ValueError(f"审核台账第{line_number}行无效: {exc}") from exc
        if decision.example_id not in known_example_ids:
            raise ValueError(f"审核台账引用未知样本: {decision.example_id}")
        rows.append(decision)
    seen = [decision.example_id for decision in rows]
    if len(seen) != len(set(seen)):
        raise ValueError("同一样本只能有一条最终审核结论")
    return rows


def build_review_packet(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    packet: list[dict[str, Any]] = []
    for record in records:
        packet.append(
            {
                "example_id": record["id"],
                "question": record["input"]["question"],
                "redis_version": record["input"].get("redis_version"),
                "evidence": record.get("evidence", []),
                "answer": record.get("parsed") or record.get("raw_output", ""),
                "review_fields": {
                    "answer_correct": None,
                    "citation_entails": None,
                    "abstention_correct": None,
                    "version_error": None,
                    "hallucination": None,
                    "notes": "",
                },
            }
        )
    return packet


def build_ablation_review_packet(ablation: dict[str, Any], limit: int = 100) -> tuple[list[dict[str, Any]], dict[str, str]]:
    systems = list(ablation["systems"].items())
    per_system, remainder = divmod(limit, len(systems))
    selected: list[tuple[str, dict[str, Any]]] = []
    for index, (system_name, system) in enumerate(systems):
        selected.extend((system_name, case) for case in system["cases"][:per_system + (index < remainder)])
    records = [
        {
            "id": f"review-{index:03d}",
            "input": case["input"],
            "evidence": case["evidence"],
            "parsed": case.get("parsed"),
            "raw_output": case.get("raw_model_output", ""),
        }
        for index, (_, case) in enumerate(selected, start=1)
    ]
    mapping = {f"review-{index:03d}": system for index, (system, _) in enumerate(selected, start=1)}
    return build_review_packet(records), mapping


def build_system_review_packet(ablation: dict[str, Any], system_name: str, start: int, limit: int) -> tuple[list[dict[str, Any]], dict[str, str]]:
    cases = ablation["systems"][system_name]["cases"][:limit]
    records = [
        {"id": f"review-{index:03d}", "input": case["input"], "evidence": case["evidence"], "parsed": case.get("parsed"), "raw_output": case.get("raw_model_output", "")}
        for index, case in enumerate(cases, start=start)
    ]
    mapping = {record["id"]: system_name for record in records}
    return build_review_packet(records), mapping


def import_completed_packet(packet: list[dict[str, Any]], reviewer_id: str) -> list[ReviewDecision]:
    decisions: list[ReviewDecision] = []
    for row in packet:
        fields = row["review_fields"]
        if any(fields[name] is None for name in ("answer_correct", "citation_entails", "abstention_correct", "version_error", "hallucination")):
            raise ValueError(f"审核未完成: {row['example_id']}")
        decisions.append(
            ReviewDecision(
                example_id=row["example_id"], reviewer_id=reviewer_id,
                answer_correct=fields["answer_correct"], citation_entails=fields["citation_entails"],
                abstention_correct=fields["abstention_correct"], version_error=fields["version_error"],
                hallucination=fields["hallucination"], notes=fields["notes"],
            )
        )
    return decisions


def summarize_decisions(decisions: list[ReviewDecision], system_map: dict[str, str]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[ReviewDecision]] = {}
    for decision in decisions:
        grouped.setdefault(system_map[decision.example_id], []).append(decision)
    summary: dict[str, dict[str, float | int]] = {}
    for system, rows in grouped.items():
        total = len(rows)
        summary[system] = {
            "reviewed_count": total,
            "human_correct_rate": sum(row.answer_correct for row in rows) / total,
            "human_citation_entailment_rate": sum(row.citation_entails for row in rows) / total,
            "human_correct_abstention_rate": sum(row.abstention_correct for row in rows) / total,
            "human_version_error_rate": sum(row.version_error for row in rows) / total,
            "human_hallucination_rate": sum(row.hallucination for row in rows) / total,
        }
    return summary


def write_review_packet(records_path: Path, output: Path, limit: int = 100) -> None:
    payload = json.loads(records_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        packet = build_review_packet(payload)
    elif isinstance(payload, dict) and "systems" in payload:
        packet, mapping = build_ablation_review_packet(payload, limit=limit)
        output.with_name(f"{output.stem}.system-map.json").write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    else:
        raise ValueError("输入必须是评测记录数组或ablation.json")
    output.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="导出RedisDoctor盲审包")
    parser.add_argument("records", type=Path, help="消融结果中的单系统records JSON文件")
    parser.add_argument("output", type=Path, help="盲审包JSON输出路径")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    write_review_packet(args.records, args.output, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
