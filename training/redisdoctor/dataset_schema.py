from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator


Split = Literal["train", "validation", "test"]
ReviewStatus = Literal["draft", "reviewed", "rejected"]


class InterviewExample(BaseModel):
    example_id: str = Field(min_length=1)
    source_group: str = Field(min_length=1)
    source_url: HttpUrl
    scenario: Literal[
        "version", "persistence", "cluster", "performance", "acl", "command", "refusal", "citation_trap"
    ]
    split: Split
    question: str = Field(min_length=12)
    evidence_source_ids: list[str] = Field(default_factory=list)
    expected_abstain: bool
    expected_citation_ids: list[str] = Field(default_factory=list)
    review_status: ReviewStatus = "draft"

    @model_validator(mode="after")
    def evidence_matches_expected_behavior(self) -> "InterviewExample":
        if self.expected_abstain and self.expected_citation_ids:
            raise ValueError("拒答样本不能要求引用")
        if not self.expected_abstain and not self.expected_citation_ids:
            raise ValueError("可回答样本必须要求至少一个引用")
        if not self.expected_abstain and not self.evidence_source_ids:
            raise ValueError("可回答样本必须提供证据来源")
        return self


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower())


def validate_source_isolation(examples: list[InterviewExample]) -> None:
    ids = [example.example_id for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("example_id必须唯一")
    groups: dict[str, set[str]] = {}
    questions: dict[str, set[str]] = {}
    for example in examples:
        groups.setdefault(example.source_group, set()).add(example.split)
        questions.setdefault(normalize_question(example.question), set()).add(example.split)
    leaking_groups = sorted(group for group, splits in groups.items() if len(splits) > 1)
    if leaking_groups:
        raise ValueError(f"source_group跨切分泄漏: {', '.join(leaking_groups)}")
    leaked_questions = sorted(question for question, splits in questions.items() if len(splits) > 1)
    if leaked_questions:
        raise ValueError("同一问题不能同时进入训练与独立测试")
