from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from repomedic.audit import AuditSnapshot, AuditTrail
from repomedic.dashscope import (
    PROMPT_VERSION,
    AgentRunError,
    DashScopeBackend,
    DashScopeClient,
    ModelAPIError,
)
from repomedic.evaluation import (
    CaseResult,
    calculate_metrics,
    load_cases,
    load_complex_cases,
    verify_fixture,
)
from repomedic.flow import RepoMedicComplexFlow, RepoMedicFlow, run_single_agent_tool_loop
from repomedic.models import DiagnosisInput, DiagnosisReport, TraceEvent

Mode = Literal["multi", "single"]
Dataset = Literal["simple", "complex"]
AUDIT_SCHEMA_VERSION = 1


class BenchmarkRunCase(BaseModel):
    case_id: str
    expected_file: str
    input: dict[str, Any]
    report: DiagnosisReport
    audit: AuditSnapshot

    def as_case_result(self) -> CaseResult:
        return CaseResult(
            case_id=self.case_id,
            expected_file=self.expected_file,
            report=self.report,
        )


def _enrich_report(report: DiagnosisReport, audit: AuditSnapshot) -> DiagnosisReport:
    return report.model_copy(
        update={
            "attempted_tool_calls": audit.attempted_tool_calls,
            "invalid_tool_calls": audit.invalid_tool_calls,
            "token_usage": audit.token_usage,
        }
    )


def _error_code(exc: Exception) -> str:
    if isinstance(exc, (AgentRunError, ModelAPIError)):
        return exc.code
    if isinstance(exc, ValidationError):
        return "schema_validation_error"
    if isinstance(exc, json.JSONDecodeError):
        return "invalid_json"
    if isinstance(exc, TimeoutError):
        return "timeout"
    return "internal_error"


def _error_report(
    request: DiagnosisInput,
    exc: Exception,
    elapsed_ms: float,
    audit: AuditSnapshot,
) -> DiagnosisReport:
    code = _error_code(exc)
    return DiagnosisReport(
        issue=request.issue,
        hypotheses=[],
        evidence=[],
        verifications=[],
        trace=[
            TraceEvent(
                role="system",
                action=f"案例失败：{code}：{exc}",
                duration_ms=elapsed_ms,
                token_usage=audit.token_usage,
            )
        ],
        elapsed_ms=elapsed_ms,
        token_usage=audit.token_usage,
        tool_calls=audit.attempted_tool_calls - audit.invalid_tool_calls,
        attempted_tool_calls=audit.attempted_tool_calls,
        invalid_tool_calls=audit.invalid_tool_calls,
        status="error",
        failure_reason=code,
    )


def _result_row(row: BenchmarkRunCase) -> dict[str, object]:
    report = row.report
    top = report.top_hypothesis
    verified = any(
        decision.confirmed
        and decision.hypothesis_index < len(report.hypotheses)
        and report.hypotheses[decision.hypothesis_index].root_file == row.expected_file
        for decision in report.verifications
    )
    return {
        "case_id": row.case_id,
        "expected_file": row.expected_file,
        "input": row.input,
        "top1_file": top.root_file if top else None,
        "top1_correct": bool(top and top.root_file == row.expected_file),
        "verified_correct_root_cause": verified,
        "failure_reason": report.failure_reason,
        "report": report.model_dump(mode="json"),
        "audit": row.audit.model_dump(mode="json"),
    }


def _write_output(
    path: Path,
    *,
    model: str,
    fixture_checks: list[dict[str, object]],
    results: dict[Mode, list[BenchmarkRunCase]],
    max_tokens_per_case: int,
    max_tokens_total: int,
    dataset: Dataset,
) -> None:
    payload: dict[str, object] = {
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(),
            "model": model,
            "temperature": 0,
            "enable_thinking": False,
            "api_timeout_seconds": 60,
            "tool_timeout_seconds": 30,
            "max_total_tokens_per_case": max_tokens_per_case,
            "max_total_tokens_for_run": max_tokens_total,
            "dataset": dataset,
            "prompt_version": PROMPT_VERSION,
            "audit_schema_version": AUDIT_SCHEMA_VERSION,
            "warning": "真实API评测；结果只适用于本固定故障集和当前配置",
        },
        "fixture_checks": fixture_checks,
    }
    for mode, rows in results.items():
        case_results = [row.as_case_result() for row in rows]
        payload[mode] = {
            "metrics": calculate_metrics(case_results).model_dump(),
            "cases": [_result_row(row) for row in rows],
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_results(
    path: Path, model: str, modes: tuple[Mode, ...]
) -> dict[Mode, list[BenchmarkRunCase]]:
    results: dict[Mode, list[BenchmarkRunCase]] = {mode: [] for mode in modes}
    if not path.exists():
        return results
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    if metadata.get("model") != model:
        raise RuntimeError(
            f"检查点模型为{metadata.get('model')!r}，当前模型为{model!r}"
        )
    if metadata.get("prompt_version") != PROMPT_VERSION:
        raise RuntimeError("检查点Prompt版本与当前版本不同，请使用新的输出文件")
    if metadata.get("audit_schema_version") != AUDIT_SCHEMA_VERSION:
        raise RuntimeError("检查点审计Schema版本与当前版本不同，请使用新的输出文件")
    for mode in modes:
        results[mode] = [
            BenchmarkRunCase.model_validate(row)
            for row in payload.get(mode, {}).get("cases", [])
        ]
    return results


def run(
    cases_path: Path,
    output: Path,
    modes: tuple[Mode, ...],
    *,
    max_tokens_per_case: int,
    max_tokens_total: int,
    dataset: Dataset,
    case_ids: set[str] | None = None,
) -> None:
    cases = load_complex_cases(cases_path) if dataset == "complex" else load_cases(cases_path)
    if case_ids is not None:
        known_ids = {case.id for case in cases}
        unknown_ids = case_ids - known_ids
        if unknown_ids:
            raise ValueError(f"未找到案例：{', '.join(sorted(unknown_ids))}")
        cases = [case for case in cases if case.id in case_ids]
    model = os.environ.get("DASHSCOPE_MODEL", "")
    client = DashScopeClient.from_env()
    results = _load_results(output, model, modes)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        repositories = {case.id: case.materialize(root) for case in cases}
        checks = [verify_fixture(case, repositories[case.id]) for case in cases]
        fixture_rows = [check.model_dump() for check in checks]
        invalid = [check.case_id for check in checks if not check.reproduced]
        if invalid:
            _write_output(
                output,
                model=model,
                fixture_checks=fixture_rows,
                results=results,
                max_tokens_per_case=max_tokens_per_case,
                max_tokens_total=max_tokens_total,
                dataset=dataset,
            )
            raise RuntimeError(f"以下故障样例未能复现，已停止API评测：{', '.join(invalid)}")

        for case in cases:
            request = DiagnosisInput(
                repository=repositories[case.id],
                issue=case.issue,
                traceback=case.traceback,
            )
            case_input: dict[str, Any] = {
                "issue": case.issue,
                "traceback": case.traceback,
                "expected_file": case.expected_file,
                "files": case.files,
                "verification_command": case.verification_command,
            }
            for mode in modes:
                if any(row.case_id == case.id for row in results[mode]):
                    print(f"[{mode}] {case.id}: checkpoint", flush=True)
                    continue
                spent_tokens = sum(
                    row.report.token_usage
                    for rows in results.values()
                    for row in rows
                )
                remaining_tokens = max_tokens_total - spent_tokens
                if remaining_tokens <= 0:
                    raise RuntimeError(
                        f"已达到整次评测Token上限：{max_tokens_total}"
                    )
                audit = AuditTrail(PROMPT_VERSION)
                backend = DashScopeBackend(
                    client,
                    audit=audit,
                    max_total_tokens=min(max_tokens_per_case, remaining_tokens),
                )
                started = time.perf_counter()
                try:
                    if mode == "multi":
                        flow = (
                            RepoMedicComplexFlow(
                                backend=backend, max_total_tokens=min(
                                    max_tokens_per_case, remaining_tokens
                                )
                            )
                            if dataset == "complex"
                            else RepoMedicFlow(backend=backend)
                        )
                        report = flow.run(request)
                    else:
                        report = run_single_agent_tool_loop(request, backend)
                    snapshot = audit.snapshot()
                    report = _enrich_report(report, snapshot)
                except Exception as exc:  # noqa: BLE001 - 单案例失败不能终止整批评测
                    audit.record(
                        phase="system",
                        kind="error",
                        success=False,
                        payload={
                            "error_type": _error_code(exc),
                            "python_type": type(exc).__name__,
                            "message": str(exc),
                        },
                    )
                    snapshot = audit.snapshot()
                    report = _error_report(
                        request,
                        exc,
                        (time.perf_counter() - started) * 1000,
                        snapshot,
                    )
                results[mode].append(
                    BenchmarkRunCase(
                        case_id=case.id,
                        expected_file=case.expected_file,
                        input=case_input,
                        report=report,
                        audit=snapshot,
                    )
                )
                _write_output(
                    output,
                    model=model,
                    fixture_checks=fixture_rows,
                    results=results,
                    max_tokens_per_case=max_tokens_per_case,
                    max_tokens_total=max_tokens_total,
                    dataset=dataset,
                )
                print(
                    f"[{mode}] {case.id}: status={report.status}; "
                    f"top1={report.top_hypothesis.root_file if report.top_hypothesis else 'none'}",
                    flush=True,
                )


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="运行RepoMedic百炼真实模型对照评测")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).with_name("cases.json"),
    )
    parser.add_argument(
        "--dataset",
        choices=("simple", "complex"),
        default="simple",
        help="simple 使用原有夹具；complex 使用 Complex-12 夹具",
    )
    parser.add_argument(
        "--max-tokens-per-case",
        type=int,
        default=35_000,
        help="单个案例与模式的输入加输出Token上限",
    )
    parser.add_argument(
        "--max-tokens-total",
        type=int,
        default=840_000,
        help="整次评测Token上限；已完成案例会从检查点恢复",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        help="只运行指定案例；可重复传入，用于低成本冒烟验证",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("multi", "single", "both"),
        default="both",
    )
    args = parser.parse_args()
    if args.dataset == "complex" and args.cases == Path(__file__).with_name("cases.json"):
        args.cases = Path(__file__).with_name("complex_cases.json")
    modes: tuple[Mode, ...] = (
        ("multi", "single") if args.mode == "both" else (args.mode,)
    )
    if args.max_tokens_per_case <= 0 or args.max_tokens_total <= 0:
        parser.error("Token上限必须为正整数")
    run(
        args.cases,
        args.output,
        modes,
        max_tokens_per_case=args.max_tokens_per_case,
        max_tokens_total=args.max_tokens_total,
        dataset=args.dataset,
        case_ids=set(args.case_id) if args.case_id else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
