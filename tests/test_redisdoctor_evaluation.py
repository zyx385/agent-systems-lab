from __future__ import annotations

import unittest

from redisdoctor.evaluation import Prediction, calculate_metrics, run_ablation
from redisdoctor.models import (
    Citation,
    EvaluationCase,
    RetrievedChunk,
    RouteDecision,
    SupportAnswer,
    SupportQuery,
)


def make_answer() -> SupportAnswer:
    route = RouteDecision(
        intent="persistence",
        module="persistence",
        rewritten_query="AOF rewrite Redis 7.2",
        mode="hybrid",
        confidence=0.9,
    )
    context = RetrievedChunk(
        source_id="aof-7",
        title="AOF",
        text="AOF rewrite compacts the log.",
        source_url="https://redis.io/aof",
        redis_version="7.2",
        module="persistence",
        score=2,
    )
    return SupportAnswer(
        answer="AOF rewrite compacts the log.",
        citations=[Citation(source_id="aof-7", source_url="https://redis.io/aof")],
        route=route,
        contexts=[context],
        latency_ms=10,
        token_usage=20,
    )


class RedisDoctorEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.case = EvaluationCase(
            id="aof",
            query=SupportQuery(text="Why is AOF rewrite slow?", redis_version="7.2"),
            expected_intent="persistence",
            relevant_source_ids=["aof-7"],
            expected_version="7.2",
        )

    def test_perfect_prediction_scores_one(self) -> None:
        metrics = calculate_metrics([Prediction(case=self.case, answer=make_answer())])
        self.assertEqual(metrics.intent_macro_f1, 1)
        self.assertEqual(metrics.recall_at_5, 1)
        self.assertEqual(metrics.mrr, 1)
        self.assertEqual(metrics.citation_precision, 1)
        self.assertEqual(metrics.hallucination_rate, 0)

    def test_ablation_requires_exact_four_systems(self) -> None:
        with self.assertRaises(ValueError):
            run_ablation([self.case], {"rag": lambda _case: make_answer()})
        systems = {name: (lambda _case: make_answer()) for name in ("base", "qlora", "rag", "qlora_rag")}
        results = run_ablation([self.case], systems)
        self.assertEqual([result.system_name for result in results], ["base", "qlora", "rag", "qlora_rag"])

    def test_unanswerable_case_rewards_refusal_and_penalizes_answer(self) -> None:
        refusal_case = EvaluationCase(
            id="unknown",
            query=SupportQuery(text="Does Redis support quantum indexes?", redis_version="7.2"),
            expected_intent="unknown",
            relevant_source_ids=[],
            expected_version="7.2",
            expected_abstained=True,
        )
        refused = make_answer().model_copy(
            update={"answer": "没有足够证据。", "citations": [], "abstained": True}
        )
        answered = make_answer().model_copy(update={"abstained": False})

        refusal_metrics = calculate_metrics([Prediction(case=refusal_case, answer=refused)])
        answer_metrics = calculate_metrics([Prediction(case=refusal_case, answer=answered)])

        self.assertEqual(refusal_metrics.correct_abstention_rate, 1)
        self.assertEqual(refusal_metrics.hallucination_rate, 0)
        self.assertEqual(answer_metrics.correct_abstention_rate, 0)
        self.assertEqual(answer_metrics.hallucination_rate, 1)

    def test_answerable_case_counts_over_refusal(self) -> None:
        refused = make_answer().model_copy(
            update={"answer": "没有足够证据。", "citations": [], "abstained": True}
        )
        metrics = calculate_metrics([Prediction(case=self.case, answer=refused)])
        self.assertEqual(metrics.over_refusal_rate, 1)
        self.assertEqual(metrics.answer_accuracy_rate, 0)

    def test_missing_context_is_not_a_version_match_for_answerable_case(self) -> None:
        refused = make_answer().model_copy(
            update={
                "answer": "没有足够证据。",
                "citations": [],
                "contexts": [],
                "abstained": True,
            }
        )
        metrics = calculate_metrics([Prediction(case=self.case, answer=refused)])
        self.assertEqual(metrics.version_match_rate, 0)


if __name__ == "__main__":
    unittest.main()
