from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from repomedic.flow import RepoMedicFlow, run_single_agent_baseline
from repomedic.models import DiagnosisInput


class RepoMedicFlowTest(unittest.TestCase):
    def test_flow_produces_evidence_backed_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pricing.py").write_text(
                "def total_price(left, right):\n    return left + right\n", encoding="utf-8"
            )
            request = DiagnosisInput(
                repository=root,
                issue="total_price concatenates price strings",
                traceback='File "pricing.py", line 2, in total_price',
            )
            report = RepoMedicFlow().run(request)
            self.assertTrue(report.hypotheses)
            self.assertEqual(report.top_hypothesis.root_file, "pricing.py")
            self.assertFalse(report.verifications[0].confirmed)
            self.assertIn("未确认", report.verifications[0].notes)
            markdown = report.render_markdown()
            self.assertIn("# RepoMedic 诊断报告", markdown)
            self.assertIn("## 证据", markdown)
            self.assertIn("## 验证结果", markdown)
            self.assertIn("## 执行轨迹", markdown)
            self.assertEqual([event.role for event in report.trace], ["triage", "investigator", "verifier"])
            self.assertGreater(report.tool_calls, 0)

    def test_baseline_uses_one_investigation_branch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.py").write_text("timeout = 30\n", encoding="utf-8")
            request = DiagnosisInput(repository=root, issue="timeout config is ignored")
            report = run_single_agent_baseline(request)
            self.assertEqual(report.trace[0].role, "baseline")
            self.assertLessEqual(report.tool_calls, 2)

    def test_report_refuses_when_no_evidence_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = DiagnosisInput(repository=root, issue="symbol_does_not_exist")
            report = RepoMedicFlow().run(request)

        self.assertEqual(report.status, "refused")
        self.assertEqual(report.failure_reason, "insufficient_evidence")


if __name__ == "__main__":
    unittest.main()
