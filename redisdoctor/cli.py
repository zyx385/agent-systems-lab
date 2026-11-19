from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .datasets import load_documents
from .generation import OpenAICompatibleGenerator
from .models import RedisDocument, SupportQuery
from .parsers import parse_redis_markdown
from .pipeline import RedisDoctorPipeline
from .retrieval import LexicalRetriever
from .routing import RuleBasedRouter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="具备版本感知能力的Redis技术支持助手")
    parser.add_argument("question", help="Redis故障、配置或排障问题")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--document", type=Path, action="append", help="Redis Markdown文档，可重复传入"
    )
    source.add_argument("--corpus", type=Path, help="经过校验的Redis官方来源JSON语料")
    parser.add_argument("--version", help="目标Redis版本，例如7.2")
    parser.add_argument("--module", default="general", help="文档所属模块，默认general")
    parser.add_argument(
        "--generator",
        choices=("extractive", "openai"),
        default="extractive",
        help="回答生成器：离线抽取或OpenAI兼容模型",
    )
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:8080/v1", help="OpenAI兼容服务地址"
    )
    parser.add_argument("--model", default="local", help="OpenAI兼容服务中的模型名称")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="API密钥环境变量名")
    parser.add_argument("--timeout", type=int, default=60, help="模型请求超时秒数")
    parser.add_argument("--minimum-score", type=float, default=4.0, help="最低证据分数")
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    documents: list[RedisDocument]
    if args.corpus:
        documents = load_documents(args.corpus)
    else:
        documents = []
        for index, path in enumerate(args.document, start=1):
            documents.extend(
                parse_redis_markdown(
                    path.read_text(encoding="utf-8"),
                    source_url=path.resolve().as_uri(),
                    source_prefix=f"doc-{index}",
                    redis_version=args.version,
                    module=args.module,
                )
            )
    generator = None
    if args.generator == "openai":
        generator = OpenAICompatibleGenerator(
            args.endpoint,
            args.model,
            os.getenv(args.api_key_env, ""),
            timeout=args.timeout,
        )
    pipeline = RedisDoctorPipeline(
        RuleBasedRouter(),
        LexicalRetriever(documents),
        generator=generator,
        minimum_score=args.minimum_score,
    )
    answer = pipeline.answer(SupportQuery(text=args.question, redis_version=args.version))
    print(json.dumps(answer.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
