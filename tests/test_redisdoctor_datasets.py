from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from redisdoctor.datasets import DatasetValidationError, load_benchmark, load_documents


class RedisDoctorDatasetTest(unittest.TestCase):
    def write_json(self, directory: Path, name: str, value: object) -> Path:
        path = directory / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_rejects_duplicate_document_ids(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            documents = [
                {
                    "source_id": "same",
                    "title": "One",
                    "text": "Document one",
                    "source_url": "https://redis.io/docs/one",
                },
                {
                    "source_id": "same",
                    "title": "Two",
                    "text": "Document two",
                    "source_url": "https://redis.io/docs/two",
                },
            ]
            cases = []
            corpus_path = self.write_json(directory, "corpus.json", documents)
            cases_path = self.write_json(directory, "cases.json", cases)
            with self.assertRaises(DatasetValidationError):
                load_benchmark(corpus_path, cases_path)

    def test_rejects_non_official_source_url(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            documents = [
                {
                    "source_id": "bad",
                    "title": "Bad",
                    "text": "Unofficial text",
                    "source_url": "https://example.com/redis",
                }
            ]
            corpus_path = self.write_json(directory, "corpus.json", documents)
            cases_path = self.write_json(directory, "cases.json", [])
            with self.assertRaises(DatasetValidationError):
                load_benchmark(corpus_path, cases_path)

    def test_rejects_case_with_unknown_relevant_source(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            documents = [
                {
                    "source_id": "known",
                    "title": "Known",
                    "text": "Known Redis fact",
                    "source_url": "https://redis.io/docs/known",
                }
            ]
            cases = [
                {
                    "id": "case",
                    "query": {"text": "What is this Redis fact?"},
                    "expected_intent": "unknown",
                    "relevant_source_ids": ["missing"],
                }
            ]
            corpus_path = self.write_json(directory, "corpus.json", documents)
            cases_path = self.write_json(directory, "cases.json", cases)
            with self.assertRaises(DatasetValidationError):
                load_benchmark(corpus_path, cases_path)

    def test_project_benchmark_is_fixed_and_valid(self) -> None:
        documents, cases = load_benchmark(
            Path("benchmarks/redisdoctor/corpus.json"),
            Path("benchmarks/redisdoctor/cases.json"),
        )
        self.assertEqual(len(documents), 12)
        self.assertEqual(len(cases), 17)
        self.assertEqual(sum(case.expected_abstained for case in cases), 3)
        self.assertEqual(load_documents(Path("benchmarks/redisdoctor/corpus.json")), documents)
