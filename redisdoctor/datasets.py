from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

from pydantic import ValidationError

from .models import EvaluationCase, RedisDocument


class DatasetValidationError(ValueError):
    pass


def _read_json_list(path: Path) -> list[object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"无法读取数据集 {path}: {exc}") from exc
    if not isinstance(value, list):
        raise DatasetValidationError(f"数据集顶层必须是数组: {path}")
    return value


def _ensure_unique(values: list[str], label: str) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise DatasetValidationError(f"{label}重复: {', '.join(duplicates)}")


def load_documents(corpus_path: Path) -> list[RedisDocument]:
    try:
        documents = [RedisDocument.model_validate(row) for row in _read_json_list(corpus_path)]
    except ValidationError as exc:
        raise DatasetValidationError(str(exc)) from exc
    _ensure_unique([document.source_id for document in documents], "source_id")
    for document in documents:
        host = (urlparse(document.source_url).hostname or "").lower()
        if host not in {"redis.io", "www.redis.io"}:
            raise DatasetValidationError(
                f"非Redis官方来源: {document.source_id} -> {document.source_url}"
            )
    return documents


def load_benchmark(
    corpus_path: Path, cases_path: Path
) -> tuple[list[RedisDocument], list[EvaluationCase]]:
    try:
        documents = load_documents(corpus_path)
        cases = [EvaluationCase.model_validate(row) for row in _read_json_list(cases_path)]
    except ValidationError as exc:
        raise DatasetValidationError(str(exc)) from exc

    _ensure_unique([case.id for case in cases], "case id")

    known_source_ids = {document.source_id for document in documents}
    for case in cases:
        unknown = sorted(set(case.relevant_source_ids) - known_source_ids)
        if unknown:
            raise DatasetValidationError(
                f"案例 {case.id} 引用了不存在的来源: {', '.join(unknown)}"
            )
        if not case.expected_abstained and not case.relevant_source_ids:
            raise DatasetValidationError(f"可回答案例 {case.id} 必须标注相关来源")
    return documents, cases
