from __future__ import annotations

import re
from typing import Protocol

from .models import (
    CoordinatorRoute,
    DiagnosisInput,
    DiagnosisPlan,
    Evidence,
    Hypothesis,
    RoleFinding,
    SharedCaseState,
    SpecialistRole,
    Verification,
)
from .tools import RepositoryTools


class DiagnosisBackend(Protocol):
    def triage(self, request: DiagnosisInput) -> tuple[DiagnosisPlan, int]: ...

    def investigate(
        self, request: DiagnosisInput, plan: DiagnosisPlan, search_term: str, tools: RepositoryTools
    ) -> tuple[list[Evidence], list[Hypothesis], int]: ...

    def verify(
        self,
        request: DiagnosisInput,
        hypotheses: list[Hypothesis],
        evidence: list[Evidence],
        tools: RepositoryTools,
    ) -> tuple[list[Verification], int]: ...


class ComplexDiagnosisBackend(DiagnosisBackend, Protocol):
    def route_complex(
        self, request: DiagnosisInput
    ) -> tuple[CoordinatorRoute, int]: ...

    def investigate_role(
        self,
        request: DiagnosisInput,
        route: CoordinatorRoute,
        state: SharedCaseState,
        role: SpecialistRole,
        tools: RepositoryTools,
    ) -> tuple[RoleFinding, list[Evidence], int]: ...


class RuleBasedBackend:
    """用于测试和透明基线的离线后端，不用于冒充LLM结果。"""

    _WORDS = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
    _TRACE_FILE = re.compile(r'File "[^"]*[\\/]([^"\\/]+\.py)", line (\d+)')

    def triage(self, request: DiagnosisInput) -> tuple[DiagnosisPlan, int]:
        words = [word.lower() for word in self._WORDS.findall(request.issue + " " + request.traceback)]
        stop = {"error", "failed", "python", "traceback", "line", "with", "from", "when"}
        terms = list(dict.fromkeys(word for word in words if word not in stop))[:6]
        if not terms:
            terms = ["raise", "return"]
        return DiagnosisPlan(search_terms=terms, rationale="从问题描述和报错证据中提取关键词"), 0

    def investigate(
        self, request: DiagnosisInput, plan: DiagnosisPlan, search_term: str, tools: RepositoryTools
    ) -> tuple[list[Evidence], list[Hypothesis], int]:
        evidence = tools.search(search_term, limit=8)
        trace_match = self._TRACE_FILE.search(request.traceback)
        if trace_match:
            filename, line = trace_match.groups()
            matches = [item for item in evidence if item.path.endswith(filename)]
            if not matches:
                matches = tools.search(filename.removesuffix(".py"), limit=4)
                evidence.extend(matches)
        hypotheses: list[Hypothesis] = []
        for item in evidence[:2]:
            hypotheses.append(
                Hypothesis(
                    summary=f"故障可能与证据位置附近的 `{search_term}` 有关。",
                    root_file=item.path,
                    confidence=0.55,
                    evidence_ids=[item.id],
                )
            )
        return evidence, hypotheses, 0

    def verify(
        self,
        request: DiagnosisInput,
        hypotheses: list[Hypothesis],
        evidence: list[Evidence],
        tools: RepositoryTools,
    ) -> tuple[list[Verification], int]:
        evidence_ids = {item.id for item in evidence}
        return (
            [
                Verification(
                    hypothesis_index=index,
                    confirmed=False,
                    notes=(
                        "所有引用的证据ID均存在，但离线后端没有运行针对该假设的测试；"
                        "根因仍未确认。"
                    ),
                )
                for index, hypothesis in enumerate(hypotheses)
                if all(item in evidence_ids for item in hypothesis.evidence_ids)
            ],
            0,
        )
