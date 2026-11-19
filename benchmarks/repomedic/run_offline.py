from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from repomedic.evaluation import CaseResult, calculate_metrics, load_cases
from repomedic.flow import RepoMedicFlow, run_single_agent_baseline
from repomedic.models import DiagnosisInput


def run(cases_path: Path) -> dict[str, object]:
    cases = load_cases(cases_path)
    multi_results: list[CaseResult] = []
    baseline_results: list[CaseResult] = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for case in cases:
            repository = case.materialize(root)
            request = DiagnosisInput(
                repository=repository, issue=case.issue, traceback=case.traceback
            )
            multi_results.append(
                CaseResult(
                    case_id=case.id,
                    expected_file=case.expected_file,
                    report=RepoMedicFlow().run(request),
                )
            )
            baseline_results.append(
                CaseResult(
                    case_id=case.id,
                    expected_file=case.expected_file,
                    report=run_single_agent_baseline(request),
                )
            )
    return {
        "warning": "离线启发式结果仅用于冒烟基线，不代表模型效果",
        "multi_agent": calculate_metrics(multi_results).model_dump(),
        "single_agent": calculate_metrics(baseline_results).model_dump(),
    }


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="运行RepoMedic的12例离线冒烟评测")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).with_name("cases.json"),
        help="故障样本JSON文件",
    )
    parser.add_argument("--output", type=Path, help="可选的评测结果JSON输出路径")
    args = parser.parse_args()
    payload = json.dumps(run(args.cases), ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
