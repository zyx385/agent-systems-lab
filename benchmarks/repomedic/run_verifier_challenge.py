from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from repomedic.audit import AuditTrail
from repomedic.dashscope import PROMPT_VERSION, DashScopeBackend
from repomedic.models import DiagnosisInput, Evidence, Hypothesis
from repomedic.tools import RepositoryTools


def run(output: Path) -> bool:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        files = {
            "app/calculator.py": "def divide(value, total):\n    return value / total\n",
            "app/formatting.py": "def format_label(value):\n    return str(value).strip()\n",
            "tests/test_calculator.py": (
                "from app.calculator import divide\n\n"
                "def test_zero_total():\n    assert divide(5, 0) == 0\n"
            ),
        }
        for relative, content in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        request = DiagnosisInput(
            repository=root,
            issue="divide crashes when total is zero",
            traceback='File "app/calculator.py", line 2, in divide',
        )
        evidence = [
            Evidence(
                id="wrong-evidence",
                kind="source",
                path="app/formatting.py",
                line=1,
                excerpt="def format_label(value):",
            )
        ]
        hypotheses = [
            Hypothesis(
                summary="错误候选：字符串格式化导致除零异常",
                root_file="app/formatting.py",
                confidence=0.8,
                evidence_ids=["wrong-evidence"],
            )
        ]
        audit = AuditTrail(PROMPT_VERSION)
        backend = DashScopeBackend.from_env(audit=audit)
        error: dict[str, str] | None = None
        try:
            verifications, tokens = backend.verify(
                request, hypotheses, evidence, RepositoryTools(root)
            )
        except Exception as exc:  # noqa: BLE001 - 挑战脚本也必须保存结构化失败
            verifications = []
            tokens = audit.snapshot().token_usage
            error = {
                "failure_reason": getattr(exc, "code", type(exc).__name__),
                "message": str(exc),
            }
        payload = {
            "metadata": {
                "model": os.environ.get("DASHSCOPE_MODEL", ""),
                "prompt_version": PROMPT_VERSION,
                "purpose": "验证Verifier会否定与失败测试无关的错误根因",
            },
            "input": {
                "issue": request.issue,
                "traceback": request.traceback,
                "files": files,
            },
            "wrong_hypothesis": hypotheses[0].model_dump(mode="json"),
            "verifications": [item.model_dump(mode="json") for item in verifications],
            "token_usage": tokens,
            "audit": audit.snapshot().model_dump(mode="json"),
            "error": error,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return error is None


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="运行Verifier错误根因否定挑战")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    succeeded = run(args.output)
    print(args.output)
    return 0 if succeeded else 2


if __name__ == "__main__":
    raise SystemExit(main())
