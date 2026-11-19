from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from repomedic.evaluation import (
    CaseResult,
    calculate_metrics,
    load_cases,
    load_complex_cases,
    verify_fixture,
)
from repomedic.models import DiagnosisReport


class RepoMedicEvaluationTest(unittest.TestCase):
    def test_complex_cases_are_unique_and_reproducible(self) -> None:
        path = (
            Path(__file__).parents[1]
            / "benchmarks"
            / "repomedic"
            / "complex_cases.json"
        )
        cases = load_complex_cases(path)
        self.assertEqual(len(cases), 12)
        self.assertEqual(len({case.id for case in cases}), 12)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checks = [verify_fixture(case, case.materialize(root)) for case in cases]
        self.assertEqual([check.case_id for check in checks if not check.reproduced], [])

    def test_complex_cases_require_cross_module_evidence(self) -> None:
        path = (
            Path(__file__).parents[1]
            / "benchmarks"
            / "repomedic"
            / "complex_cases.json"
        )
        cases = load_complex_cases(path)
        self.assertTrue(all(len(case.source_files) >= 3 for case in cases))
        self.assertTrue(all(len(case.required_questions) >= 2 for case in cases))
        self.assertTrue(all(len(case.suggested_roles) >= 2 for case in cases))

    def test_checked_in_benchmark_has_twelve_unique_cases(self) -> None:
        path = Path(__file__).parents[1] / "benchmarks" / "repomedic" / "cases.json"
        cases = load_cases(path)
        self.assertEqual(len(cases), 12)
        self.assertEqual(len({case.id for case in cases}), 12)

    def test_all_checked_in_faults_are_reproducible(self) -> None:
        path = Path(__file__).parents[1] / "benchmarks" / "repomedic" / "cases.json"
        cases = load_cases(path)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checks = [verify_fixture(case, case.materialize(root)) for case in cases]
        failures = [check.case_id for check in checks if not check.reproduced]
        self.assertEqual(failures, [])

    def test_empty_metrics_are_deterministic(self) -> None:
        metrics = calculate_metrics([])
        self.assertEqual(metrics.case_count, 0)
        self.assertEqual(metrics.top1_location_accuracy, 0)

    def test_metrics_include_invalid_tools_and_refusals(self) -> None:
        report = DiagnosisReport(
            issue="unknown",
            hypotheses=[],
            evidence=[],
            verifications=[],
            trace=[],
            elapsed_ms=1,
            tool_calls=1,
            attempted_tool_calls=2,
            invalid_tool_calls=1,
            status="refused",
            failure_reason="insufficient_evidence",
        )
        metrics = calculate_metrics(
            [CaseResult(case_id="case", expected_file="app.py", report=report)]
        )

        self.assertEqual(metrics.invalid_tool_rate, 0.5)
        self.assertEqual(metrics.refusal_rate, 1)
        self.assertEqual(metrics.error_rate, 0)


if __name__ == "__main__":
    unittest.main()
