from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="离线回放 RepoMedic 已验证运行记录")
    parser.add_argument("log", type=Path)
    parser.add_argument("--mode", choices=("multi", "single"))
    parser.add_argument("--case-id")
    args = parser.parse_args()
    payload = json.loads(args.log.read_text(encoding="utf-8"))
    row = payload.get("case", payload)
    if args.mode:
        rows = payload[args.mode]["cases"]
        row = next(item for item in rows if item["case_id"] == args.case_id)
    print(f"输入：{row['input']['issue']}")
    for event in row["audit"]["events"]:
        if event["kind"] == "tool_call":
            print(
                f"模型调用：{event['payload']['name']} "
                f"{event['payload']['arguments_raw']}"
            )
        elif event["kind"] == "tool_result":
            status = "成功" if event["success"] else "拒绝"
            print(f"工具结果：{status} · {event['payload']['name']}")
    report = row["report"]
    top = report["hypotheses"][0] if report["hypotheses"] else None
    print(f"最终状态：{report['status']}")
    print(f"Top-1：{top['root_file'] if top else '无'}")
    confirmed = any(item["confirmed"] for item in report["verifications"])
    print(f"Verifier确认：{'是' if confirmed else '否'}")
    print(f"Token：{report['token_usage']}；耗时：{report['elapsed_ms']:.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
