from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


Complexity = Literal["simple", "complex"]
SpecialistRole = Literal[
    "call_chain", "security", "state", "config", "async", "test_forensics"
]


class DiagnosisInput(BaseModel):
    repository: Path
    issue: str = Field(min_length=3)
    traceback: str = ""

    @field_validator("repository")
    @classmethod
    def repository_must_exist(cls, value: Path) -> Path:
        resolved = value.resolve()
        if not resolved.is_dir():
            raise ValueError("repository 必须是一个已存在的目录")
        return resolved


class Evidence(BaseModel):
    id: str
    kind: Literal["source", "traceback", "test", "diff", "static"]
    path: str
    line: int | None = None
    excerpt: str
    command: list[str] = Field(default_factory=list)


class CoordinatorRoute(BaseModel):
    complexity: Complexity
    reason: str = Field(min_length=3)
    roles: list[SpecialistRole] = Field(default_factory=list)
    candidate_files: list[str] = Field(min_length=1)
    evidence_questions: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_route(self) -> CoordinatorRoute:
        if len(set(self.roles)) != len(self.roles):
            raise ValueError("专项角色不能重复")
        if self.complexity == "simple" and self.roles:
            raise ValueError("简单案例不能选择专项角色")
        if self.complexity == "complex" and len(self.roles) < 2:
            raise ValueError("复杂案例至少需要两个专项角色")
        return self


class RoleFinding(BaseModel):
    role: SpecialistRole
    question: str = Field(min_length=3)
    evidence_ids: list[str] = Field(min_length=1)
    candidate_files: list[str] = Field(min_length=1)
    primary_file: str = Field(min_length=3)
    claim: str = Field(min_length=3)
    confidence: float = Field(ge=0, le=1)
    open_questions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def primary_file_must_be_candidate(self) -> RoleFinding:
        if self.primary_file not in self.candidate_files:
            raise ValueError("主要根因文件必须位于候选文件中")
        return self


class RoleFailure(BaseModel):
    role: SpecialistRole
    code: Literal[
        "invalid_json",
        "schema_validation_error",
        "token_limit",
        "tool_error",
        "api_timeout",
        "internal_error",
    ]
    message: str


class SharedCaseState(BaseModel):
    evidence: list[Evidence] = Field(default_factory=list)
    evidence_roles: dict[str, list[SpecialistRole]] = Field(default_factory=dict)
    full_file_reads: set[str] = Field(default_factory=set)
    findings: list[RoleFinding] = Field(default_factory=list)
    failures: list[RoleFailure] = Field(default_factory=list)

    def add_evidence(
        self, items: list[Evidence], role: SpecialistRole
    ) -> list[Evidence]:
        existing = {(item.path, item.line, item.excerpt) for item in self.evidence}
        accepted: list[Evidence] = []
        for item in items:
            key = (item.path, item.line, item.excerpt)
            if key in existing:
                continue
            existing.add(key)
            accepted.append(item)
            self.evidence.append(item)
            self.evidence_roles.setdefault(item.id, []).append(role)
        return accepted

    def record_full_file_read(self, path: str) -> bool:
        if path in self.full_file_reads:
            return False
        self.full_file_reads.add(path)
        return True


class Hypothesis(BaseModel):
    summary: str
    root_file: str
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(min_length=1)
    suggested_patch: str = ""


class Verification(BaseModel):
    hypothesis_index: int = Field(ge=0)
    confirmed: bool
    commands: list[list[str]] = Field(default_factory=list)
    notes: str


class TraceEvent(BaseModel):
    role: Literal["triage", "investigator", "verifier", "system", "baseline"]
    action: str
    duration_ms: float = Field(ge=0)
    token_usage: int = Field(default=0, ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DiagnosisPlan(BaseModel):
    search_terms: list[str] = Field(min_length=1, max_length=8)
    test_commands: list[list[str]] = Field(default_factory=list)
    rationale: str


class DiagnosisReport(BaseModel):
    issue: str
    hypotheses: list[Hypothesis]
    evidence: list[Evidence]
    verifications: list[Verification]
    trace: list[TraceEvent]
    elapsed_ms: float = Field(ge=0)
    token_usage: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    attempted_tool_calls: int = Field(default=0, ge=0)
    invalid_tool_calls: int = Field(default=0, ge=0)
    status: Literal["completed", "refused", "error"] = "completed"
    failure_reason: str = ""

    @property
    def top_hypothesis(self) -> Hypothesis | None:
        return self.hypotheses[0] if self.hypotheses else None

    def render_markdown(self) -> str:
        lines = [
            "# RepoMedic 诊断报告",
            "",
            f"**问题描述：** {self.issue}",
            f"**状态：** {self.status}",
            "",
        ]
        if self.failure_reason:
            lines.extend([f"**失败或拒答原因：** `{self.failure_reason}`", ""])
        if not self.hypotheses:
            lines.append("未生成有证据支持的候选根因。")
        for index, hypothesis in enumerate(self.hypotheses, start=1):
            lines.extend(
                [
                    f"## 候选根因 {index}：{hypothesis.root_file}",
                    "",
                    hypothesis.summary,
                    "",
                    f"置信度：`{hypothesis.confidence:.2f}`",
                    f"证据引用：{', '.join(hypothesis.evidence_ids)}",
                ]
            )
            if hypothesis.suggested_patch:
                lines.extend(["", "建议补丁：", "```diff", hypothesis.suggested_patch, "```"])
            lines.append("")
        lines.extend(["## 证据", ""])
        if not self.evidence:
            lines.append("- 未检索到证据。")
        for item in self.evidence:
            location = f"{item.path}:{item.line}" if item.line is not None else item.path
            lines.append(f"- `{item.id}` · `{location}` · {item.excerpt}")
        lines.extend(["", "## 验证结果", ""])
        if not self.verifications:
            lines.append("- 尚未执行验证。")
        for verification in self.verifications:
            status = "已确认" if verification.confirmed else "未确认"
            lines.append(
                f"- 候选根因 {verification.hypothesis_index + 1}：**{status}**。"
                f"{verification.notes}"
            )
            for command in verification.commands:
                lines.append(f"  - 命令：`{' '.join(command)}`")
        lines.extend(["", "## 执行轨迹", ""])
        for event in self.trace:
            lines.append(
                f"- `{event.role}`：{event.action}；{event.duration_ms:.1f} ms；"
                f"Token {event.token_usage}"
            )
        lines.extend(
            [
                "",
                "## 运行统计",
                "",
                f"- 总耗时：{self.elapsed_ms:.1f} ms",
                f"- 工具调用次数：{self.tool_calls}",
                f"- 模型尝试工具次数：{self.attempted_tool_calls}",
                f"- 无效工具次数：{self.invalid_tool_calls}",
                f"- Token用量：{self.token_usage}",
            ]
        )
        return "\n".join(lines)
