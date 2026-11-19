from __future__ import annotations

import time
from typing import Protocol

from pydantic import BaseModel, Field

from .citations import has_sufficient_evidence, validate_citations
from .models import RetrievedChunk, SupportAnswer, SupportQuery
from .retrieval import Retriever
from .routing import QueryRouter


class GeneratedResponse(BaseModel):
    answer: str
    cited_source_ids: list[str] = Field(default_factory=list)
    token_usage: int = Field(default=0, ge=0)


class GenerationError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


class AnswerGenerator(Protocol):
    def generate(
        self, query: SupportQuery, contexts: list[RetrievedChunk]
    ) -> GeneratedResponse: ...


class ExtractiveGenerator:
    """展示证据约束行为的离线生成器，不用于冒充LLM结果。"""

    def generate(
        self, query: SupportQuery, contexts: list[RetrievedChunk]
    ) -> GeneratedResponse:
        if not contexts:
            return GeneratedResponse(answer="", cited_source_ids=[])
        context = contexts[0]
        excerpt = " ".join(context.text.split())[:700]
        return GeneratedResponse(
            answer=f"根据《{context.title}》中的证据：{excerpt}",
            cited_source_ids=[context.source_id],
        )


class RedisDoctorPipeline:
    def __init__(
        self,
        router: QueryRouter,
        retriever: Retriever,
        generator: AnswerGenerator | None = None,
        minimum_score: float = 1.0,
    ) -> None:
        self.router = router
        self.retriever = retriever
        self.generator = generator or ExtractiveGenerator()
        self.minimum_score = minimum_score

    def answer(self, query: SupportQuery) -> SupportAnswer:
        started = time.perf_counter()
        route, route_tokens = self.router.route(query)
        contexts = self.retriever.retrieve(query, route)
        if not has_sufficient_evidence(contexts, self.minimum_score):
            return SupportAnswer(
                answer="没有足够且与Redis版本兼容的证据，无法安全回答该问题。",
                abstained=True,
                reason="insufficient_evidence",
                route=route,
                contexts=contexts,
                latency_ms=(time.perf_counter() - started) * 1000,
                token_usage=route_tokens,
            )
        try:
            generated = self.generator.generate(query, contexts)
        except GenerationError as exc:
            return SupportAnswer(
                answer="回答生成服务暂时无法使用，已安全停止，未输出未经验证的结论。",
                abstained=True,
                reason="generation_error",
                error_code=exc.code,
                route=route,
                contexts=contexts,
                latency_ms=(time.perf_counter() - started) * 1000,
                token_usage=route_tokens,
            )
        citations, invalid = validate_citations(generated.cited_source_ids, contexts)
        if invalid or not citations:
            return SupportAnswer(
                answer="生成内容无法通过检索来源验证，因此拒绝给出结论。",
                abstained=True,
                reason="invalid_citations",
                route=route,
                contexts=contexts,
                latency_ms=(time.perf_counter() - started) * 1000,
                token_usage=route_tokens + generated.token_usage,
            )
        return SupportAnswer(
            answer=generated.answer,
            citations=citations,
            route=route,
            contexts=contexts,
            latency_ms=(time.perf_counter() - started) * 1000,
            token_usage=route_tokens + generated.token_usage,
        )
