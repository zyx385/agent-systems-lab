from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from .agents import ComplexDiagnosisBackend, DiagnosisBackend, RuleBasedBackend
from .models import (
    DiagnosisInput,
    DiagnosisReport,
    Evidence,
    Hypothesis,
    RoleFailure,
    SharedCaseState,
    TraceEvent,
    Verification,
)
from .tools import RepositoryTools


class SharedRoleTools:
    """在不改变安全策略的前提下，阻止角色重复读取共享的完整文件。"""

    def __init__(self, delegate: RepositoryTools, state: SharedCaseState) -> None:
        self._delegate = delegate
        self._state = state

    @property
    def call_count(self) -> int:
        return self._delegate.call_count

    def search(self, pattern: str, limit: int = 30):
        return self._delegate.search(pattern, limit)

    def read(self, relative_path: str, start: int = 1, end: int = 200) -> str:
        if start == 1 and end >= 200 and not self._state.record_full_file_read(relative_path):
            raise ValueError(f"共享证据已包含完整文件：{relative_path}")
        return self._delegate.read(relative_path, start, end)

    def call_graph(self, relative_path: str):
        return self._delegate.call_graph(relative_path)

    def run(self, command: list[str]):
        return self._delegate.run(command)


class SingleAgentBackend(Protocol):
    def diagnose_single_agent(
        self, request: DiagnosisInput, tools: RepositoryTools
    ) -> tuple[list[Evidence], list[Hypothesis], list[Verification], int]: ...


class RepoMedicFlow:
    def __init__(self, backend: DiagnosisBackend | None = None, max_workers: int = 3) -> None:
        self.backend = backend or RuleBasedBackend()
        self.max_workers = max_workers

    def run(self, request: DiagnosisInput) -> DiagnosisReport:
        started = time.perf_counter()
        trace: list[TraceEvent] = []
        tools = RepositoryTools(request.repository)

        stage = time.perf_counter()
        plan, triage_tokens = self.backend.triage(request)
        trace.append(
            TraceEvent(
                role="triage",
                action="创建诊断计划",
                duration_ms=(time.perf_counter() - stage) * 1000,
                token_usage=triage_tokens,
            )
        )

        evidence: list[Evidence] = []
        hypotheses: list[Hypothesis] = []
        stage = time.perf_counter()
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(plan.search_terms))) as pool:
            futures = [
                pool.submit(self.backend.investigate, request, plan, term, tools)
                for term in plan.search_terms
            ]
            investigation_tokens = 0
            for future in futures:
                found_evidence, found_hypotheses, tokens = future.result()
                evidence.extend(found_evidence)
                hypotheses.extend(found_hypotheses)
                investigation_tokens += tokens
        evidence = self._deduplicate_evidence(evidence)
        hypotheses = self._rank_hypotheses(hypotheses, evidence)
        trace.append(
            TraceEvent(
                role="investigator",
                action=f"并行调查 {len(plan.search_terms)} 条检索分支",
                duration_ms=(time.perf_counter() - stage) * 1000,
                token_usage=investigation_tokens,
            )
        )

        stage = time.perf_counter()
        verifications, verifier_tokens = self.backend.verify(request, hypotheses, evidence, tools)
        trace.append(
            TraceEvent(
                role="verifier",
                action="独立验证证据和候选根因",
                duration_ms=(time.perf_counter() - stage) * 1000,
                token_usage=verifier_tokens,
            )
        )
        elapsed = (time.perf_counter() - started) * 1000
        return DiagnosisReport(
            issue=request.issue,
            hypotheses=hypotheses,
            evidence=evidence,
            verifications=verifications,
            trace=trace,
            elapsed_ms=elapsed,
            token_usage=sum(item.token_usage for item in trace),
            tool_calls=tools.call_count,
            attempted_tool_calls=tools.call_count,
            status="completed" if hypotheses else "refused",
            failure_reason="" if hypotheses else "insufficient_evidence",
        )

    @staticmethod
    def _deduplicate_evidence(items: list[Evidence]) -> list[Evidence]:
        unique: dict[tuple[str, int | None, str], Evidence] = {}
        for item in items:
            key = (item.path, item.line, item.excerpt)
            unique.setdefault(key, item)
        normalized: list[Evidence] = []
        for index, item in enumerate(unique.values(), start=1):
            normalized.append(item.model_copy(update={"id": f"evidence-{index}"}))
        return normalized

    @staticmethod
    def _rank_hypotheses(items: list[Hypothesis], evidence: list[Evidence]) -> list[Hypothesis]:
        by_location = {(item.path, item.line): item.id for item in evidence}
        ranked: dict[str, Hypothesis] = {}
        for hypothesis in items:
            evidence_ids = [
                evidence_id
                for (path, _line), evidence_id in by_location.items()
                if path == hypothesis.root_file
            ]
            if not evidence_ids:
                continue
            updated = hypothesis.model_copy(update={"evidence_ids": evidence_ids})
            current = ranked.get(updated.root_file)
            if current is None or updated.confidence > current.confidence:
                ranked[updated.root_file] = updated
        return sorted(ranked.values(), key=lambda item: (-item.confidence, item.root_file))


class RepoMedicComplexFlow:
    """按复杂度调度专项角色；角色失败不会阻断独立验证。"""

    def __init__(self, backend: ComplexDiagnosisBackend, max_total_tokens: int = 35_000) -> None:
        self.backend = backend
        self.max_total_tokens = max_total_tokens

    def run(self, request: DiagnosisInput) -> DiagnosisReport:
        started = time.perf_counter()
        tools = RepositoryTools(request.repository)
        trace: list[TraceEvent] = []
        stage = time.perf_counter()
        route, route_tokens = self.backend.route_complex(request)
        trace.append(
            TraceEvent(
                role="triage",
                action=f"复杂度路由：{route.complexity}",
                duration_ms=(time.perf_counter() - stage) * 1000,
                token_usage=route_tokens,
            )
        )
        if route.complexity == "simple":
            return run_single_agent_tool_loop(request, self.backend)

        state = SharedCaseState()
        role_tokens = 0
        for role in route.roles:
            stage = time.perf_counter()
            try:
                finding, evidence, tokens = self.backend.investigate_role(
                    request, route, state, role, SharedRoleTools(tools, state)
                )
                state.add_evidence(evidence, role)
                if finding not in state.findings:
                    state.findings.append(finding)
                role_tokens += tokens
                trace.append(
                    TraceEvent(
                        role="investigator",
                        action=f"专项角色完成：{role}；{finding.claim}",
                        duration_ms=(time.perf_counter() - stage) * 1000,
                        token_usage=tokens,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - 单角色失败必须隔离
                code = getattr(exc, "code", "internal_error")
                if code not in RoleFailure.model_fields["code"].annotation.__args__:
                    code = "internal_error"
                state.failures.append(RoleFailure(role=role, code=code, message=str(exc)))
                trace.append(
                    TraceEvent(
                        role="system",
                        action=f"专项角色失败：{role}：{code}",
                        duration_ms=(time.perf_counter() - stage) * 1000,
                    )
                )

        hypotheses = self._hypotheses_from_findings(state)
        stage = time.perf_counter()
        verifications, verifier_tokens = self.backend.verify(
            request, hypotheses, state.evidence, tools
        )
        trace.append(
            TraceEvent(
                role="verifier",
                action="独立验证专项角色候选根因",
                duration_ms=(time.perf_counter() - stage) * 1000,
                token_usage=verifier_tokens,
            )
        )
        elapsed = (time.perf_counter() - started) * 1000
        return DiagnosisReport(
            issue=request.issue,
            hypotheses=hypotheses,
            evidence=state.evidence,
            verifications=verifications,
            trace=trace,
            elapsed_ms=elapsed,
            token_usage=route_tokens + role_tokens + verifier_tokens,
            tool_calls=tools.call_count,
            attempted_tool_calls=tools.call_count,
            status="completed" if hypotheses else ("error" if state.failures else "refused"),
            failure_reason="" if hypotheses else (state.failures[0].code if state.failures else "insufficient_evidence"),
        )

    @staticmethod
    def _hypotheses_from_findings(state: SharedCaseState) -> list[Hypothesis]:
        by_file: dict[str, Hypothesis] = {}
        for finding in state.findings:
            path = finding.primary_file
            current = by_file.get(path)
            candidate = Hypothesis(
                summary=finding.claim,
                root_file=path,
                confidence=finding.confidence,
                evidence_ids=finding.evidence_ids,
            )
            if current is None or candidate.confidence > current.confidence:
                by_file[path] = candidate
        return sorted(by_file.values(), key=lambda item: (-item.confidence, item.root_file))


def run_single_agent_baseline(
    request: DiagnosisInput, backend: DiagnosisBackend | None = None
) -> DiagnosisReport:
    """使用相同后端和工具面的单检索词基线。"""
    backend = backend or RuleBasedBackend()
    tools = RepositoryTools(request.repository)
    started = time.perf_counter()
    plan, triage_tokens = backend.triage(request)
    evidence, hypotheses, investigation_tokens = backend.investigate(
        request, plan, plan.search_terms[0], tools
    )
    verifications, verifier_tokens = backend.verify(request, hypotheses, evidence, tools)
    elapsed = (time.perf_counter() - started) * 1000
    return DiagnosisReport(
        issue=request.issue,
        hypotheses=hypotheses,
        evidence=evidence,
        verifications=verifications,
        trace=[
            TraceEvent(
                role="baseline",
                action="单Agent诊断",
                duration_ms=elapsed,
                token_usage=triage_tokens + investigation_tokens + verifier_tokens,
            )
        ],
        elapsed_ms=elapsed,
        token_usage=triage_tokens + investigation_tokens + verifier_tokens,
        tool_calls=tools.call_count,
        attempted_tool_calls=tools.call_count,
        status="completed" if hypotheses else "refused",
        failure_reason="" if hypotheses else "insufficient_evidence",
    )


def run_single_agent_tool_loop(
    request: DiagnosisInput, backend: SingleAgentBackend
) -> DiagnosisReport:
    """用一个模型角色和一个连续工具循环完成诊断。"""
    tools = RepositoryTools(request.repository)
    started = time.perf_counter()
    evidence, hypotheses, verifications, tokens = backend.diagnose_single_agent(request, tools)
    elapsed = (time.perf_counter() - started) * 1000
    return DiagnosisReport(
        issue=request.issue,
        hypotheses=hypotheses,
        evidence=evidence,
        verifications=verifications,
        trace=[
            TraceEvent(
                role="baseline",
                action="单Agent工具循环诊断",
                duration_ms=elapsed,
                token_usage=tokens,
            )
        ],
        elapsed_ms=elapsed,
        token_usage=tokens,
        tool_calls=tools.call_count,
        attempted_tool_calls=tools.call_count,
        status="completed" if hypotheses else "refused",
        failure_reason="" if hypotheses else "insufficient_evidence",
    )
