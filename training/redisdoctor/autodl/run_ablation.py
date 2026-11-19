from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from redisdoctor.datasets import load_documents
from redisdoctor.retrieval import version_compatible


class RunPlan(BaseModel):
    corpus: Path
    cases: Path
    adapter: Path
    output: Path
    model_name: str = "Qwen/Qwen3-4B"
    retriever_model_name: str = "BAAI/bge-m3"
    top_k: int = Field(default=5, ge=1, le=10)
    max_new_tokens: int = Field(default=384, ge=32, le=1024)
    seed: int = 20261109


class ModelAnswer(BaseModel):
    intent: str
    module: str
    rewritten_query: str
    mode: str
    should_abstain: bool
    reason: str
    answer: str
    cited_source_ids: list[str]


SYSTEM_PROMPT = (
    "你是RedisDoctor。只能依据evidence回答Redis问题。只输出JSON对象，字段必须是"
    "intent、module、rewritten_query、mode、should_abstain、reason、answer、cited_source_ids。"
    "若证据不足，should_abstain必须为true且cited_source_ids必须为空。"
)


def validate_plan(plan: RunPlan, *, overwrite: bool = False, resume: bool = False) -> None:
    for label, path in (("语料", plan.corpus), ("测试集", plan.cases), ("Adapter", plan.adapter)):
        if not path.exists():
            raise ValueError(f"{label}不存在: {path}")
    if plan.output.exists() and not (overwrite or resume):
        raise ValueError(f"结果目录已存在，拒绝覆盖: {plan.output}")


def _extract_json(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1]).strip()
    start = stripped.find("{")
    if start < 0:
        raise ValueError("模型输出未包含JSON对象")
    value, _ = json.JSONDecoder().raw_decode(stripped[start:])
    if not isinstance(value, dict):
        raise TypeError("模型输出JSON不是对象")
    return value


class VectorRetriever:
    def __init__(self, documents: list[Any], model_name: str) -> None:
        try:
            import faiss  # type: ignore[import-not-found]
            import numpy as np  # type: ignore[import-not-found]
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("缺少远端依赖，请先运行bootstrap.sh") from exc
        self._faiss = faiss
        self._np = np
        self.documents = documents
        self.encoder = SentenceTransformer(model_name, device="cpu")
        vectors = self.encoder.encode(
            [f"{document.title}\n{document.text}" for document in documents],
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(np.asarray(vectors, dtype="float32"))

    def retrieve(
        self, question: str, version: str | None, top_k: int, *, version_filter: bool, minimum_score: float | None
    ) -> list[dict[str, Any]]:
        query = self.encoder.encode([question], normalize_embeddings=True, show_progress_bar=False)
        scores, indices = self.index.search(self._np.asarray(query, dtype="float32"), top_k * 3)
        rows: list[dict[str, Any]] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            document = self.documents[int(index)]
            if version_filter and not version_compatible(document.redis_version, version):
                continue
            if minimum_score is not None and float(score) < minimum_score:
                continue
            rows.append(
                {
                    "source_id": document.source_id,
                    "title": document.title,
                    "text": document.text,
                    "source_url": document.source_url,
                    "redis_version": document.redis_version,
                    "score": float(score),
                }
            )
            if len(rows) == top_k:
                break
        return rows


class LocalGenerator:
    def __init__(self, model_name: str, adapter: Path | None, max_new_tokens: int) -> None:
        try:
            import torch  # type: ignore[import-not-found]
            from peft import PeftModel  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise RuntimeError("缺少远端模型依赖，请先运行bootstrap.sh") from exc
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization,
        )
        self.model = PeftModel.from_pretrained(model, str(adapter)) if adapter else model
        self.model.eval()

    def generate(self, question: str, version: str | None, evidence: list[dict[str, Any]]) -> tuple[str, int]:
        payload = {"question": question, "redis_version": version, "evidence": evidence}
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        output_tokens = generated.shape[-1] - inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(generated[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True), int(output_tokens)

    def close(self) -> None:
        del self.model
        self.torch.cuda.empty_cache()


def _safe_answer(
    raw: str, evidence: list[dict[str, Any]], *, citation_validation: bool
) -> tuple[ModelAnswer | None, str]:
    try:
        answer = ModelAnswer.model_validate(_extract_json(raw))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return None, f"invalid_json: {exc}"
    available = {row["source_id"] for row in evidence}
    if citation_validation and not set(answer.cited_source_ids).issubset(available):
        return None, "invalid_citations"
    if answer.should_abstain and answer.cited_source_ids:
        return None, "abstention_with_citations"
    return answer, ""


def _metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {}
    total = len(records)
    answerable = [row for row in records if not row["expected"]["should_abstain"]]
    abstain_cases = [row for row in records if row["expected"]["should_abstain"]]
    retrieval = [row for row in records if row["evidence"]]
    recall = [
        float(bool(set(row["expected"]["cited_source_ids"]) & set(row["retrieved_source_ids"])))
        for row in answerable
    ]
    mrr = []
    for row in answerable:
        rank = next(
            (
                index
                for index, source_id in enumerate(row["retrieved_source_ids"], start=1)
                if source_id in set(row["expected"]["cited_source_ids"])
            ),
            0,
        )
        mrr.append(1 / rank if rank else 0.0)
    correct_answer = [
        float(
            row["parsed"] is not None
            and not row["parsed"]["should_abstain"]
            and bool(set(row["parsed"]["cited_source_ids"]) & set(row["expected"]["cited_source_ids"]))
        )
        for row in answerable
    ]
    correct_abstain = [
        float(row["parsed"] is not None and row["parsed"]["should_abstain"])
        for row in abstain_cases
    ]
    grounded = [
        float(
            row["parsed"] is not None
            and set(row["parsed"]["cited_source_ids"]).issubset(set(row["retrieved_source_ids"]))
        )
        for row in records
    ]
    return {
        "case_count": float(total),
        "retrieval_case_count": float(len(retrieval)),
        "recall_at_5": sum(recall) / len(recall) if recall else 0.0,
        "mrr": sum(mrr) / len(mrr) if mrr else 0.0,
        "answer_accuracy_rate": sum(correct_answer) / len(correct_answer) if correct_answer else 0.0,
        "correct_abstention_rate": sum(correct_abstain) / len(correct_abstain) if correct_abstain else 0.0,
        "grounded_output_rate": sum(grounded) / len(grounded) if grounded else 0.0,
        "valid_json_rate": sum(row["parsed"] is not None for row in records) / total,
        "error_rate": sum(bool(row["error"]) for row in records) / total,
        "mean_latency_ms": sum(row["latency_ms"] for row in records) / total,
        "mean_generated_tokens": sum(row["generated_tokens"] for row in records) / total,
    }


def _run_system(
    name: str,
    generator: LocalGenerator,
    cases: list[dict[str, Any]],
    retriever: VectorRetriever | None,
    top_k: int,
    *,
    version_filter: bool,
    citation_validation: bool,
    minimum_score: float | None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for case in cases:
        payload = case["input"]
        evidence = (
            retriever.retrieve(
                payload["question"], payload.get("redis_version"), top_k,
                version_filter=version_filter, minimum_score=minimum_score,
            )
            if retriever else []
        )
        started = time.perf_counter()
        raw = ""
        generated_tokens = 0
        error = ""
        parsed: ModelAnswer | None = None
        try:
            raw, generated_tokens = generator.generate(
                payload["question"], payload.get("redis_version"), evidence
            )
            parsed, error = _safe_answer(raw, evidence, citation_validation=citation_validation)
        except (OSError, RuntimeError, ValueError) as exc:
            error = f"generation_error: {type(exc).__name__}: {exc}"
        records.append(
            {
                "id": case["id"],
                "source_group": case["source_group"],
                "input": payload,
                "expected": case["expected"],
                "evidence": evidence,
                "retrieved_source_ids": [row["source_id"] for row in evidence],
                "raw_model_output": raw,
                "parsed": parsed.model_dump(mode="json") if parsed else None,
                "error": error,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "generated_tokens": generated_tokens,
            }
        )
    return {"system": name, "metrics": _metrics(records), "cases": records}


def _environment() -> dict[str, Any]:
    payload: dict[str, Any] = {"python": sys.version, "platform": platform.platform()}
    try:
        import torch

        payload["torch"] = torch.__version__
        payload["cuda"] = torch.version.cuda
        payload["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"
    except ImportError:
        payload["torch"] = "unavailable"
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行RedisDoctor QLoRA × 向量RAG四组消融")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--retriever-model", default="BAAI/bge-m3")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--matrix", type=Path, default=Path("training/redisdoctor/autodl/experiment_matrix.json"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    plan = RunPlan(
        corpus=args.corpus,
        cases=args.cases,
        adapter=args.adapter,
        output=args.output,
        model_name=args.model,
        retriever_model_name=args.retriever_model,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
    )
    validate_plan(plan, overwrite=args.overwrite, resume=args.resume)
    plan.output.mkdir(parents=True)
    cases = json.loads(plan.cases.read_text(encoding="utf-8"))
    documents = load_documents(plan.corpus)
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    conditions = matrix["conditions"]
    retriever = VectorRetriever(documents, plan.retriever_model_name)
    partial_path = plan.output / "ablation.partial.json"
    systems: dict[str, dict[str, Any]] = {}
    if args.resume and partial_path.exists():
        systems = json.loads(partial_path.read_text(encoding="utf-8")).get("systems", {})
    for adapter_enabled in (False, True):
        selected = [condition for condition in conditions if bool(condition["adapter"]) == adapter_enabled]
        if not selected:
            continue
        generator = LocalGenerator(plan.model_name, plan.adapter if adapter_enabled else None, plan.max_new_tokens)
        for condition in selected:
            if condition["id"] in systems:
                continue
            systems[condition["id"]] = _run_system(
                condition["id"], generator, cases, retriever if condition["rag"] else None, plan.top_k,
                version_filter=bool(condition["version_filter"]),
                citation_validation=bool(condition["citation_validation"]),
                minimum_score=condition["minimum_score"],
            )
            _write_json(partial_path, {"systems": systems})
        generator.close()
    result = {
        "schema_version": "redisdoctor-autodl-ablation-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "plan": plan.model_dump(mode="json"),
        "inputs": {"corpus_sha256": _sha256(plan.corpus), "cases_sha256": _sha256(plan.cases)},
        "experiment_matrix": matrix,
        "environment": _environment(),
        "systems": systems,
    }
    _write_json(plan.output / "ablation.json", result)
    print(json.dumps({name: row["metrics"] for name, row in systems.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
