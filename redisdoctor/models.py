from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class RedisDocument(BaseModel):
    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    redis_version: str | None = None
    module: str = "general"
    kind: Literal["documentation", "issue", "faq"] = "documentation"


class SupportQuery(BaseModel):
    text: str = Field(min_length=3)
    redis_version: str | None = None


class RouteDecision(BaseModel):
    intent: Literal[
        "configuration", "persistence", "performance", "replication", "command", "unknown"
    ]
    module: str
    rewritten_query: str
    mode: Literal["local", "global", "hybrid"]
    confidence: float = Field(ge=0, le=1)


class RetrievedChunk(BaseModel):
    source_id: str
    title: str
    text: str
    source_url: str
    redis_version: str | None = None
    module: str
    score: float = Field(ge=0)


class Citation(BaseModel):
    source_id: str
    source_url: str


class SupportAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    abstained: bool = False
    reason: str = ""
    route: RouteDecision
    contexts: list[RetrievedChunk] = Field(default_factory=list)
    latency_ms: float = Field(default=0, ge=0)
    token_usage: int = Field(default=0, ge=0)
    error_code: str = ""

    @field_validator("citations")
    @classmethod
    def citations_are_unique(cls, value: list[Citation]) -> list[Citation]:
        seen: set[str] = set()
        unique: list[Citation] = []
        for citation in value:
            if citation.source_id not in seen:
                unique.append(citation)
                seen.add(citation.source_id)
        return unique


class EvaluationCase(BaseModel):
    id: str
    query: SupportQuery
    expected_intent: str
    relevant_source_ids: list[str]
    expected_version: str | None = None
    expected_abstained: bool = False
