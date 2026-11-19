from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from statistics import mean

from pydantic import BaseModel, Field

from .models import EvaluationCase, SupportAnswer


class Prediction(BaseModel):
    case: EvaluationCase
    answer: SupportAnswer


class EvaluationMetrics(BaseModel):
    case_count: int = Field(ge=0)
    intent_macro_f1: float = Field(ge=0, le=1)
    recall_at_5: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    version_match_rate: float = Field(ge=0, le=1)
    citation_precision: float = Field(ge=0, le=1)
    grounded_answer_rate: float = Field(ge=0, le=1)
    hallucination_rate: float = Field(ge=0, le=1)
    correct_abstention_rate: float = Field(ge=0, le=1)
    over_refusal_rate: float = Field(ge=0, le=1)
    answer_accuracy_rate: float = Field(ge=0, le=1)
    mean_latency_ms: float = Field(ge=0)
    mean_token_usage: float = Field(ge=0)


class AblationResult(BaseModel):
    system_name: str
    metrics: EvaluationMetrics


def _macro_f1(expected: Sequence[str], predicted: Sequence[str]) -> float:
    labels = sorted(set(expected) | set(predicted))
    if not labels:
        return 0.0
    scores: list[float] = []
    for label in labels:
        true_positive = sum(e == label and p == label for e, p in zip(expected, predicted))
        false_positive = sum(e != label and p == label for e, p in zip(expected, predicted))
        false_negative = sum(e == label and p != label for e, p in zip(expected, predicted))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0)
    return mean(scores)


def calculate_metrics(predictions: list[Prediction]) -> EvaluationMetrics:
    if not predictions:
        return EvaluationMetrics(
            case_count=0,
            intent_macro_f1=0,
            recall_at_5=0,
            mrr=0,
            version_match_rate=0,
            citation_precision=0,
            grounded_answer_rate=0,
            hallucination_rate=0,
            correct_abstention_rate=0,
            over_refusal_rate=0,
            answer_accuracy_rate=0,
            mean_latency_ms=0,
            mean_token_usage=0,
        )
    expected_intents = [row.case.expected_intent for row in predictions]
    predicted_intents = [row.answer.route.intent for row in predictions]
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    version_matches: list[float] = []
    citation_precisions: list[float] = []
    grounded: list[float] = []
    hallucinated: list[float] = []
    correct_abstentions: list[float] = []
    over_refusals: list[float] = []
    answer_accuracies: list[float] = []
    for row in predictions:
        retrieved = [context.source_id for context in row.answer.contexts[:5]]
        relevant = set(row.case.relevant_source_ids)
        if relevant:
            recalls.append(1.0 if relevant.intersection(retrieved) else 0.0)
            rank = next(
                (index for index, source_id in enumerate(retrieved, start=1) if source_id in relevant),
                0,
            )
            reciprocal_ranks.append(1 / rank if rank else 0.0)
        if row.case.expected_version:
            if not row.answer.contexts:
                version_matches.append(float(row.case.expected_abstained))
            else:
                version_matches.append(
                    1.0
                    if all(
                        context.redis_version is None
                        or context.redis_version.split(".", 1)[0]
                        == row.case.expected_version.split(".", 1)[0]
                        for context in row.answer.contexts
                    )
                    else 0.0
                )
        else:
            version_matches.append(1.0)
        cited = {citation.source_id for citation in row.answer.citations}
        if relevant:
            citation_precisions.append(len(cited & relevant) / len(cited) if cited else 0.0)
        is_grounded = not row.answer.abstained and bool(cited) and cited.issubset(set(retrieved))
        grounded.append(float(is_grounded))
        is_correct_answer = (
            not row.case.expected_abstained
            and not row.answer.abstained
            and bool(cited & relevant)
            and cited.issubset(set(retrieved))
        )
        if row.case.expected_abstained:
            correct_abstentions.append(float(row.answer.abstained))
        else:
            over_refusals.append(float(row.answer.abstained))
            answer_accuracies.append(float(is_correct_answer))
        hallucinated.append(
            float(
                not row.answer.abstained
                and (row.case.expected_abstained or not is_correct_answer)
            )
        )

    def safe_mean(values: list[float]) -> float:
        return mean(values) if values else 0.0

    return EvaluationMetrics(
        case_count=len(predictions),
        intent_macro_f1=_macro_f1(expected_intents, predicted_intents),
        recall_at_5=safe_mean(recalls),
        mrr=safe_mean(reciprocal_ranks),
        version_match_rate=mean(version_matches),
        citation_precision=safe_mean(citation_precisions),
        grounded_answer_rate=mean(grounded),
        hallucination_rate=mean(hallucinated),
        correct_abstention_rate=safe_mean(correct_abstentions),
        over_refusal_rate=safe_mean(over_refusals),
        answer_accuracy_rate=safe_mean(answer_accuracies),
        mean_latency_ms=mean(row.answer.latency_ms for row in predictions),
        mean_token_usage=mean(row.answer.token_usage for row in predictions),
    )


def run_ablation(
    cases: list[EvaluationCase],
    systems: dict[str, Callable[[EvaluationCase], SupportAnswer]],
) -> list[AblationResult]:
    if set(systems) != {"base", "qlora", "rag", "qlora_rag"}:
        raise ValueError("systems必须且只能包含base、qlora、rag和qlora_rag")
    grouped: dict[str, list[Prediction]] = defaultdict(list)
    for name, system in systems.items():
        for case in cases:
            grouped[name].append(Prediction(case=case, answer=system(case)))
    return [
        AblationResult(system_name=name, metrics=calculate_metrics(grouped[name]))
        for name in ("base", "qlora", "rag", "qlora_rag")
    ]
