from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from statistics import mean

from pydantic import BaseModel, Field, model_validator

from .models import DiagnosisReport


COMPLEX_SPECIALIST_ROLES = frozenset(
    {"call_chain", "security", "state", "config", "async", "test_forensics"}
)


class BenchmarkCase(BaseModel):
    id: str
    issue: str
    traceback: str = ""
    expected_file: str
    files: dict[str, str]
    verification_command: list[str]

    @model_validator(mode="after")
    def expected_file_is_seeded(self) -> BenchmarkCase:
        if self.expected_file not in self.files:
            raise ValueError("expected_file必须存在于files中")
        return self

    def materialize(self, destination: Path) -> Path:
        root = destination / self.id
        for relative, content in self.files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        return root


class ComplexBenchmarkCase(BenchmarkCase):
    required_questions: list[str] = Field(min_length=2)
    suggested_roles: list[str] = Field(min_length=2)

    @property
    def source_files(self) -> list[str]:
        return [path for path in self.files if not path.startswith("tests/")]

    @model_validator(mode="after")
    def requires_cross_module_contract(self) -> ComplexBenchmarkCase:
        if len(self.source_files) < 3:
            raise ValueError("复杂案例至少需要三个非测试源码文件")
        unknown_roles = set(self.suggested_roles) - COMPLEX_SPECIALIST_ROLES
        if unknown_roles:
            raise ValueError(f"复杂案例包含未知角色：{', '.join(sorted(unknown_roles))}")
        if len(set(self.suggested_roles)) != len(self.suggested_roles):
            raise ValueError("复杂案例角色不能重复")
        return self


class CaseResult(BaseModel):
    case_id: str
    expected_file: str
    report: DiagnosisReport


class BenchmarkMetrics(BaseModel):
    case_count: int = Field(ge=0)
    top1_location_accuracy: float = Field(ge=0, le=1)
    verified_root_cause_rate: float = Field(ge=0, le=1)
    mean_latency_ms: float = Field(ge=0)
    mean_token_usage: float = Field(ge=0)
    mean_tool_calls: float = Field(ge=0)
    invalid_tool_rate: float = Field(default=0, ge=0, le=1)
    refusal_rate: float = Field(default=0, ge=0, le=1)
    error_rate: float = Field(default=0, ge=0, le=1)


class FixtureCheck(BaseModel):
    case_id: str
    reproduced: bool
    returncode: int
    output_tail: str


def load_cases(path: Path) -> list[BenchmarkCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = [BenchmarkCase.model_validate(item) for item in raw]
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("评测样本ID必须唯一")
    return cases


def load_complex_cases(path: Path) -> list[ComplexBenchmarkCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = [ComplexBenchmarkCase.model_validate(item) for item in raw]
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("复杂评测样本ID必须唯一")
    return cases


def verify_fixture(case: BenchmarkCase, repository: Path, timeout: int = 30) -> FixtureCheck:
    """执行样例声明的命令；非零返回码表示预置故障被真实复现。"""
    try:
        command = list(case.verification_command)
        if command and command[0] == "python":
            command[0] = sys.executable
        completed = subprocess.run(
            command,
            cwd=repository,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            check=False,
        )
        output = completed.stdout + completed.stderr
        unusable = "No module named pytest" in output or "NO TESTS RAN" in output
        return FixtureCheck(
            case_id=case.id,
            reproduced=completed.returncode != 0 and not unusable,
            returncode=completed.returncode,
            output_tail=output[-1000:],
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
        output = (stdout or "") + (stderr or "")
        return FixtureCheck(
            case_id=case.id,
            reproduced=False,
            returncode=124,
            output_tail=str(output)[-1000:],
        )


def calculate_metrics(results: Iterable[CaseResult]) -> BenchmarkMetrics:
    rows = list(results)
    if not rows:
        return BenchmarkMetrics(
            case_count=0,
            top1_location_accuracy=0,
            verified_root_cause_rate=0,
            mean_latency_ms=0,
            mean_token_usage=0,
            mean_tool_calls=0,
            invalid_tool_rate=0,
            refusal_rate=0,
            error_rate=0,
        )
    top1 = [
        int(
            (hypothesis := row.report.top_hypothesis) is not None
            and hypothesis.root_file == row.expected_file
        )
        for row in rows
    ]
    verified = [
        int(
            any(
                decision.confirmed
                and decision.hypothesis_index < len(row.report.hypotheses)
                and row.report.hypotheses[decision.hypothesis_index].root_file
                == row.expected_file
                for decision in row.report.verifications
            )
        )
        for row in rows
    ]
    attempted_tool_calls = sum(row.report.attempted_tool_calls for row in rows)
    invalid_tool_calls = sum(row.report.invalid_tool_calls for row in rows)
    return BenchmarkMetrics(
        case_count=len(rows),
        top1_location_accuracy=mean(top1),
        verified_root_cause_rate=mean(verified),
        mean_latency_ms=mean(row.report.elapsed_ms for row in rows),
        mean_token_usage=mean(row.report.token_usage for row in rows),
        mean_tool_calls=mean(row.report.tool_calls for row in rows),
        invalid_tool_rate=(
            invalid_tool_calls / attempted_tool_calls if attempted_tool_calls else 0
        ),
        refusal_rate=mean(row.report.status == "refused" for row in rows),
        error_rate=mean(row.report.status == "error" for row in rows),
    )
