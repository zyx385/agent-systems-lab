from __future__ import annotations

from .models import Citation, RetrievedChunk


def validate_citations(
    cited_source_ids: list[str], contexts: list[RetrievedChunk]
) -> tuple[list[Citation], list[str]]:
    available = {context.source_id: context for context in contexts}
    valid: list[Citation] = []
    invalid: list[str] = []
    for source_id in dict.fromkeys(cited_source_ids):
        context = available.get(source_id)
        if context is None:
            invalid.append(source_id)
        else:
            valid.append(Citation(source_id=source_id, source_url=context.source_url))
    return valid, invalid


def has_sufficient_evidence(contexts: list[RetrievedChunk], minimum_score: float) -> bool:
    return bool(contexts) and contexts[0].score >= minimum_score
