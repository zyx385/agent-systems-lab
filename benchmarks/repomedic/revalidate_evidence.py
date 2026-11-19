from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from repomedic.dashscope import _test_failure_supports_hypothesis
from repomedic.evaluation import CaseResult, calculate_metrics
from repomedic.models import DiagnosisReport
from repomedic.tools import CommandResult, RepositoryTools


def _materialize(root: Path, files: dict[str, str]) -> None:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _executions(events: list[dict[str, Any]]) -> list[CommandResult]:
    rows: list[CommandResult] = []
    for event in events:
        if event.get("kind") != "tool_result" or event.get("payload", {}).get("name") != "run_command":
            continue
        result = event.get("payload", {}).get("result")
        if isinstance(result, dict) and "command" in result and "returncode" in result:
            rows.append(CommandResult.model_validate(result))
    return rows


def run(source: Path, output: Path) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        for mode in ("multi", "single"):
            case_results: list[CaseResult] = []
            for row in payload[mode]["cases"]:
                root = workspace / mode / row["case_id"]
                _materialize(root, row["input"]["files"])
                report = DiagnosisReport.model_validate(row["report"])
                executions = _executions(row["audit"]["events"])
                verifications = []
                for verification in report.verifications:
                    hypothesis = (
                        report.hypotheses[verification.hypothesis_index]
                        if verification.hypothesis_index < len(report.hypotheses)
                        else None
                    )
                    confirmed = bool(
                        verification.confirmed
                        and hypothesis
                        and any(
                            _test_failure_supports_hypothesis(
                                execution, hypothesis, RepositoryTools(root)
                            )
                            for execution in executions
                        )
                    )
                    verifications.append(
                        verification.model_copy(update={"confirmed": confirmed})
                    )
                report = report.model_copy(update={"verifications": verifications})
                row["report"] = report.model_dump(mode="json")
                row["verified_correct_root_cause"] = any(
                    decision.confirmed
                    and decision.hypothesis_index < len(report.hypotheses)
                    and report.hypotheses[decision.hypothesis_index].root_file
                    == row["expected_file"]
                    for decision in report.verifications
                )
                case_results.append(
                    CaseResult(
                        case_id=row["case_id"],
                        expected_file=row["expected_file"],
                        report=report,
                    )
                )
            payload[mode]["metrics"] = calculate_metrics(case_results).model_dump()
    payload["metadata"]["verification_revalidated_with"] = (
        "pytest exit-code and failure-to-root-file binding v1"
    )
    payload["metadata"]["source_evidence_file"] = source.name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="按严格规则重新核验已保存的RepoMedic证据")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
