from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from pydantic import ValidationError

from repomedic.flow import RepoMedicComplexFlow
from repomedic.models import (
    CoordinatorRoute,
    DiagnosisInput,
    Evidence,
    RoleFinding,
    SharedCaseState,
    Verification,
)


class FakeComplexBackend:
    def route_complex(self, request):
        return (
            CoordinatorRoute(
                complexity="complex",
                reason="cross module permission state",
                roles=["call_chain", "security"],
                candidate_files=["app/access.py"],
                evidence_questions=["What passes organization scope?", "Where is scope enforced?"],
            ),
            3,
        )

    def investigate_role(self, request, route, state, role, tools):
        if role == "security":
            raise ValueError("simulated role failure")
        evidence = Evidence(
            id="source-1",
            kind="source",
            path="app/access.py",
            line=2,
            excerpt="return document",
        )
        state.add_evidence([evidence], role)
        return (
            RoleFinding(
                role=role,
                question="What passes organization scope?",
                evidence_ids=["source-1"],
                candidate_files=["app/access.py"],
                primary_file="app/access.py",
                claim="Repository call omits organization scope.",
                confidence=0.9,
            ),
            [evidence],
            5,
        )

    def verify(self, request, hypotheses, evidence, tools):
        return ([Verification(hypothesis_index=0, confirmed=True, notes="test failed")], 2)


class RepoMedicMultiAgentV2Test(unittest.TestCase):
    def test_simple_route_cannot_select_specialists(self) -> None:
        with self.assertRaises(ValidationError):
            CoordinatorRoute(
                complexity="simple",
                reason="one obvious file",
                roles=["state"],
                candidate_files=["app/a.py"],
                evidence_questions=["Which file contains the fault?"],
            )

    def test_finding_requires_role_question_and_evidence(self) -> None:
        with self.assertRaises(ValidationError):
            RoleFinding(
                role="config",
                question="Which value wins?",
                evidence_ids=[],
                candidate_files=["app/config.py"],
                primary_file="app/config.py",
                claim="defaults win",
                confidence=0.8,
            )

    def test_shared_state_deduplicates_evidence_and_full_file_reads(self) -> None:
        state = SharedCaseState()
        evidence = Evidence(
            id="source-1",
            kind="source",
            path="app/config.py",
            line=3,
            excerpt="return bool(raw)",
        )

        self.assertEqual(state.add_evidence([evidence], "config"), [evidence])
        self.assertEqual(state.add_evidence([evidence], "security"), [])
        self.assertTrue(state.record_full_file_read("app/config.py"))
        self.assertFalse(state.record_full_file_read("app/config.py"))

    def test_role_failure_does_not_prevent_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = RepoMedicComplexFlow(FakeComplexBackend()).run(
                DiagnosisInput(repository=Path(directory), issue="organization scope bypass")
            )
        self.assertEqual(report.status, "completed")
        self.assertTrue(report.verifications[0].confirmed)
        self.assertTrue(any("专项角色失败：security" in item.action for item in report.trace))


if __name__ == "__main__":
    unittest.main()
