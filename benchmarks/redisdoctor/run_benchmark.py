from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from redisdoctor.datasets import load_benchmark
from redisdoctor.evaluation import Prediction, calculate_metrics
from redisdoctor.models import EvaluationCase, RetrievedChunk, RouteDecision, SupportQuery
from redisdoctor.pipeline import ExtractiveGenerator, RedisDoctorPipeline
from redisdoctor.retrieval import LexicalRetriever
from redisdoctor.routing import RuleBasedRouter


class EmptyRetriever:
    def retrieve(
        self, query: SupportQuery, route: RouteDecision, *, limit: int = 5
    ) -> list[RetrievedChunk]:
        return []


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_system(
    cases: list[EvaluationCase], pipeline: RedisDoctorPipeline
) -> tuple[list[Prediction], list[dict[str, object]]]:
    predictions: list[Prediction] = []
    records: list[dict[str, object]] = []
    for case in cases:
        answer = pipeline.answer(case.query)
        predictions.append(Prediction(case=case, answer=answer))
        records.append(
            {
                "case": case.model_dump(mode="json"),
                "answer": answer.model_dump(mode="json"),
                "retrieved_source_ids": [context.source_id for context in answer.contexts],
            }
        )
    return predictions, records


def run_benchmark(
    corpus_path: Path,
    cases_path: Path,
    *,
    minimum_score: float,
) -> dict[str, object]:
    documents, cases = load_benchmark(corpus_path, cases_path)
    router = RuleBasedRouter()
    systems = {
        "no_rag_extractive": RedisDoctorPipeline(
            router,
            EmptyRetriever(),
            generator=ExtractiveGenerator(),
            minimum_score=minimum_score,
        ),
        "lexical_rag_extractive": RedisDoctorPipeline(
            router,
            LexicalRetriever(documents),
            generator=ExtractiveGenerator(),
            minimum_score=minimum_score,
        ),
    }
    results: dict[str, object] = {}
    for name, pipeline in systems.items():
        predictions, records = _run_system(cases, pipeline)
        results[name] = {
            "metrics": calculate_metrics(predictions).model_dump(mode="json"),
            "cases": records,
        }
    return {
        "schema_version": "redisdoctor-benchmark-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "configuration": {
            "corpus_path": str(corpus_path),
            "cases_path": str(cases_path),
            "corpus_sha256": _sha256(corpus_path),
            "cases_sha256": _sha256(cases_path),
            "document_count": len(documents),
            "case_count": len(cases),
            "answerable_case_count": sum(not case.expected_abstained for case in cases),
            "expected_refusal_count": sum(case.expected_abstained for case in cases),
            "minimum_score": minimum_score,
            "router": "RuleBasedRouter",
            "retriever": "LexicalRetriever",
            "generator": "ExtractiveGenerator",
            "llm_used": False,
            "lightrag_used": False,
            "notes": "可复现离线RAG基线；不代表LLM、LightRAG或QLoRA效果。",
        },
        "systems": results,
    }


def _summary(result: dict[str, object]) -> str:
    systems = result["systems"]
    lines = [
        "# RedisDoctor 可复现基准",
        "",
        "> 这是词法检索 + 抽取式生成的真实离线基线，不是 LLM、LightRAG 或 QLoRA 结果。",
        "",
        "| 系统 | Recall@5 | MRR | 回答准确率 | 正确拒答率 | 过度拒答率 | 幻觉率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    assert isinstance(systems, dict)
    for name, system in systems.items():
        assert isinstance(system, dict)
        metrics = system["metrics"]
        assert isinstance(metrics, dict)
        lines.append(
            f"| {name} | {metrics['recall_at_5']:.3f} | {metrics['mrr']:.3f} | "
            f"{metrics['answer_accuracy_rate']:.3f} | {metrics['correct_abstention_rate']:.3f} | "
            f"{metrics['over_refusal_rate']:.3f} | {metrics['hallucination_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "- 语料为 12 条 Redis 官方页面的人工压缩摘要，题集为 17 道人工标注诊断题。",
            "- 结果只说明这批固定题上的行为，不能外推为生产质量。",
            "- `acl-redis7-inherited-feature` 暴露了当前同主版本过滤规则的过度拒答限制。",
            "- LLM、LightRAG 与 QLoRA 未在本结果中运行。",
            "",
            "## 运行配置",
            "",
            "- 语料：`benchmarks/redisdoctor/corpus.json`，12 条 Redis 官方来源摘要。",
            "- 题集：`benchmarks/redisdoctor/cases.json`，14 道可回答题与 3 道期望拒答题。",
            "- 路由：`RuleBasedRouter`；检索：`LexicalRetriever`；生成：`ExtractiveGenerator`。",
            "- `minimum_score=4.0`，Top K 默认 5。",
            "- JSON 保存语料与题集 SHA-256、逐题输入、上下文、分数、引用和延迟。",
            "",
            "## 主要失败",
            "",
            "`acl-redis7-inherited-feature` 应由 Redis 6 引入的 ACL 文档回答，但同主版本过滤器把该文档从 Redis 7 查询中删除，最终因最高分 `3.216 < 4.0` 拒答。因此 Recall@5、MRR、引用准确率和回答准确率均为 `13/14 = 92.86%`，过度拒答率为 `1/14 = 7.14%`。",
            "",
            "当前版本元数据只能表示一个主版本，无法表示“从 6.0 引入且后续继续有效”。",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行RedisDoctor固定题集基准")
    parser.add_argument(
        "--corpus", type=Path, default=Path("benchmarks/redisdoctor/corpus.json")
    )
    parser.add_argument("--cases", type=Path, default=Path("benchmarks/redisdoctor/cases.json"))
    parser.add_argument("--minimum-score", type=float, default=4.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_benchmark(args.corpus, args.cases, minimum_score=args.minimum_score)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(_summary(result), encoding="utf-8")
    systems = result["systems"]
    assert isinstance(systems, dict)
    print(json.dumps({name: row["metrics"] for name, row in systems.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
