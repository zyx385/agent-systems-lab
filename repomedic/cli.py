from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .audit import AuditTrail
from .dashscope import PROMPT_VERSION, DashScopeBackend
from .flow import RepoMedicFlow, run_single_agent_baseline, run_single_agent_tool_loop
from .models import DiagnosisInput


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="证据优先的Python代码仓库故障诊断工具")
    parser.add_argument("repository", type=Path, help="待诊断的Python仓库路径")
    parser.add_argument("--issue", required=True, help="Issue、故障现象或错误描述")
    parser.add_argument("--traceback", default="", help="可选的Traceback、失败测试或日志")
    parser.add_argument("--baseline", action="store_true", help="运行单Agent工具循环基线")
    parser.add_argument(
        "--backend",
        choices=("rule", "dashscope"),
        default="rule",
        help="诊断后端：rule为离线基线，dashscope为百炼真实模型",
    )
    parser.add_argument(
        "--json", action="store_true", dest="as_json", help="输出保留英文字段名的JSON"
    )
    parser.add_argument(
        "--audit-log",
        type=Path,
        help="保存输入、原始模型回复、工具调用结果和最终报告的JSON审计日志",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    request = DiagnosisInput(
        repository=args.repository, issue=args.issue, traceback=args.traceback
    )
    audit = AuditTrail(PROMPT_VERSION) if args.backend == "dashscope" else None
    backend = DashScopeBackend.from_env(audit=audit) if audit else None
    try:
        if args.baseline and backend:
            report = run_single_agent_tool_loop(request, backend)
        elif args.baseline:
            report = run_single_agent_baseline(request)
        else:
            report = RepoMedicFlow(backend=backend).run(request)
        snapshot = audit.snapshot() if audit else None
        if snapshot:
            report = report.model_copy(
                update={
                    "attempted_tool_calls": snapshot.attempted_tool_calls,
                    "invalid_tool_calls": snapshot.invalid_tool_calls,
                    "token_usage": snapshot.token_usage,
                }
            )
    except Exception as exc:  # noqa: BLE001 - CLI边界必须把任意案例错误转成结构化结果
        snapshot = audit.snapshot() if audit else None
        error = {
            "status": "error",
            "failure_reason": getattr(exc, "code", type(exc).__name__),
            "message": str(exc),
        }
        if args.audit_log:
            args.audit_log.parent.mkdir(parents=True, exist_ok=True)
            args.audit_log.write_text(
                json.dumps(
                    {
                        "input": request.model_dump(mode="json"),
                        "error": error,
                        "audit": snapshot.model_dump(mode="json") if snapshot else None,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        print(json.dumps(error, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    if args.audit_log:
        args.audit_log.parent.mkdir(parents=True, exist_ok=True)
        args.audit_log.write_text(
            json.dumps(
                {
                    "input": request.model_dump(mode="json"),
                    "report": report.model_dump(mode="json"),
                    "audit": snapshot.model_dump(mode="json") if snapshot else None,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(report.model_dump_json(indent=2) if args.as_json else report.render_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
