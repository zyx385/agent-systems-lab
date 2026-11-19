from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Protocol

from .models import RedisDocument, RetrievedChunk, RouteDecision, SupportQuery


class Retriever(Protocol):
    def retrieve(
        self, query: SupportQuery, route: RouteDecision, *, limit: int = 5
    ) -> list[RetrievedChunk]: ...


_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "can",
    "do",
    "does",
    "how",
    "i",
    "if",
    "in",
    "is",
    "of",
    "or",
    "redis",
    "should",
    "the",
    "to",
    "what",
    "when",
    "which",
    "why",
    "with",
}


def _tokens(text: str) -> list[str]:
    raw_tokens = re.findall(r"[a-z0-9_.-]+|[\u4e00-\u9fff]", text.lower())
    return [
        token
        for token in raw_tokens
        if token not in _STOPWORDS and not re.fullmatch(r"\d+(?:\.\d+)*", token)
    ]


def version_compatible(document_version: str | None, requested_version: str | None) -> bool:
    if not requested_version or not document_version:
        return True
    requested_major = requested_version.split(".", 1)[0]
    document_major = document_version.split(".", 1)[0]
    return requested_major == document_major


class LexicalRetriever:
    """使用类似TF-IDF词项重叠的确定性离线基线。"""

    def __init__(self, documents: list[RedisDocument]) -> None:
        self.documents = documents
        document_frequency = Counter(
            token for document in documents for token in set(_tokens(document.title + " " + document.text))
        )
        self.idf = {
            token: math.log((len(documents) + 1) / (count + 1)) + 1
            for token, count in document_frequency.items()
        }

    def retrieve(
        self, query: SupportQuery, route: RouteDecision, *, limit: int = 5
    ) -> list[RetrievedChunk]:
        query_tokens = Counter(_tokens(route.rewritten_query))
        candidates: list[RetrievedChunk] = []
        for document in self.documents:
            if not version_compatible(document.redis_version, query.redis_version):
                continue
            document_tokens = Counter(_tokens(document.title + " " + document.text))
            lexical_score = sum(
                min(count, document_tokens[token]) * self.idf.get(token, 1.0)
                for token, count in query_tokens.items()
            )
            if lexical_score <= 0:
                continue
            score = lexical_score
            if document.module == route.module:
                score += 0.75
            candidates.append(
                RetrievedChunk(
                    source_id=document.source_id,
                    title=document.title,
                    text=document.text,
                    source_url=document.source_url,
                    redis_version=document.redis_version,
                    module=document.module,
                    score=score,
                )
            )
        return sorted(candidates, key=lambda item: (-item.score, item.source_id))[:limit]


class LightRAGRetriever:
    """LightRAG轻量适配器，并在检索后强制执行版本过滤。"""

    def __init__(self, rag: Any, metadata: dict[str, RedisDocument]) -> None:
        self.rag = rag
        self.metadata = metadata

    def retrieve(
        self, query: SupportQuery, route: RouteDecision, *, limit: int = 5
    ) -> list[RetrievedChunk]:
        try:
            from lightrag import QueryParam  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "请先安装RedisDoctor集成依赖：pip install -e '.[redisdoctor]'"
            ) from exc
        result = self.rag.query(
            route.rewritten_query,
            param=QueryParam(mode=route.mode, only_need_context=True, top_k=limit * 2),
        )
        raw_chunks = result if isinstance(result, list) else []
        chunks: list[RetrievedChunk] = []
        for index, item in enumerate(raw_chunks):
            source_id = str(item.get("source_id", ""))
            document = self.metadata.get(source_id)
            if document is None or not version_compatible(document.redis_version, query.redis_version):
                continue
            chunks.append(
                RetrievedChunk(
                    source_id=source_id,
                    title=document.title,
                    text=str(item.get("content", document.text)),
                    source_url=document.source_url,
                    redis_version=document.redis_version,
                    module=document.module,
                    score=float(item.get("score", 1 / (index + 1))),
                )
            )
        return chunks[:limit]
