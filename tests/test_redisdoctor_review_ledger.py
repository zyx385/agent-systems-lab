from __future__ import annotations

import json
import unittest
from pathlib import Path

from training.redisdoctor.review_ledger import ReviewDecision, build_ablation_review_packet, build_review_packet, import_completed_packet, load_review_ledger, summarize_decisions


class ReviewLedgerTest(unittest.TestCase):
    def test_unknown_or_duplicate_decision_is_rejected(self) -> None:
        path = Path("tests/.review-ledger-test.jsonl")
        self.addCleanup(path.unlink, missing_ok=True)
        row = {"example_id": "unknown", "reviewer_id": "u", "answer_correct": True, "citation_entails": True, "abstention_correct": True}
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "未知"):
            load_review_ledger(path, {"case-1"})

    def test_packet_excludes_system_name_and_has_review_fields(self) -> None:
        packet = build_review_packet([{"id": "case-1", "input": {"question": "q"}, "evidence": [], "parsed": {"answer": "a"}}])
        self.assertEqual(packet[0]["example_id"], "case-1")
        self.assertNotIn("system", packet[0])
        self.assertIsNone(packet[0]["review_fields"]["answer_correct"])

    def test_ablation_packet_hides_system_and_writes_mapping_separately(self) -> None:
        ablation = {"systems": {"base_rag": {"cases": [{"input": {"question": "q"}, "evidence": [], "parsed": {"answer": "a"}}]}}}
        packet, mapping = build_ablation_review_packet(ablation)
        self.assertNotIn("base_rag", json.dumps(packet))
        self.assertEqual(mapping["review-001"], "base_rag")

    def test_completed_packet_is_imported_and_summarized_by_system(self) -> None:
        packet = [{"example_id": "review-001", "review_fields": {"answer_correct": True, "citation_entails": False, "abstention_correct": True, "version_error": False, "hallucination": True, "notes": "citation misses evidence"}}]
        decisions = import_completed_packet(packet, "user")
        self.assertIsInstance(decisions[0], ReviewDecision)
        summary = summarize_decisions(decisions, {"review-001": "qlora_rag"})
        self.assertEqual(summary["qlora_rag"]["human_correct_rate"], 1.0)
        self.assertEqual(summary["qlora_rag"]["human_hallucination_rate"], 1.0)
