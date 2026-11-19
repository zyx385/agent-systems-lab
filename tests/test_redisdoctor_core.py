from __future__ import annotations

import unittest

from redisdoctor.models import RedisDocument, SupportQuery
from redisdoctor.parsers import parse_github_issues, parse_redis_markdown
from redisdoctor.pipeline import GeneratedResponse, GenerationError, RedisDoctorPipeline
from redisdoctor.retrieval import LexicalRetriever
from redisdoctor.routing import RuleBasedRouter


class InvalidCitationGenerator:
    def generate(self, query: SupportQuery, contexts: list) -> GeneratedResponse:
        return GeneratedResponse(answer="unsupported", cited_source_ids=["missing-source"])


class FailingGenerator:
    def generate(self, query: SupportQuery, contexts: list) -> GeneratedResponse:
        raise GenerationError("timeout", "local model timed out")


class RedisDoctorCoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = [
            RedisDocument(
                source_id="aof-7",
                title="AOF persistence",
                text="AOF rewrite compacts the append-only file and may increase latency.",
                source_url="https://redis.io/docs/aof-7",
                redis_version="7.2",
                module="persistence",
            ),
            RedisDocument(
                source_id="aof-6",
                title="Legacy AOF persistence",
                text="AOF rewrite behavior for Redis 6.",
                source_url="https://redis.io/docs/aof-6",
                redis_version="6.2",
                module="persistence",
            ),
        ]

    def test_markdown_parser_preserves_metadata(self) -> None:
        documents = parse_redis_markdown(
            "# Persistence\nAOF stores writes.\n## Rewrite\nRewrite compacts data.",
            source_url="https://redis.io/docs/persistence",
            source_prefix="persistence",
            redis_version="7.2",
            module="persistence",
        )
        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[1].redis_version, "7.2")
        self.assertEqual(documents[1].title, "Rewrite")

    def test_issue_parser_skips_incomplete_records(self) -> None:
        documents = parse_github_issues(
            [
                {"number": 1, "title": "Slow AOF", "body": "Details", "labels": ["performance"]},
                {"number": 2, "title": "Missing body"},
            ],
            repository_url="https://github.com/redis/redis",
        )
        self.assertEqual([document.source_id for document in documents], ["issue-1"])

    def test_retriever_filters_incompatible_major_versions(self) -> None:
        router = RuleBasedRouter()
        query = SupportQuery(text="Why is AOF rewrite slow?", redis_version="7.0")
        route, _ = router.route(query)
        chunks = LexicalRetriever(self.documents).retrieve(query, route)
        self.assertEqual([chunk.source_id for chunk in chunks], ["aof-7"])

    def test_retriever_does_not_create_evidence_from_module_boost_only(self) -> None:
        router = RuleBasedRouter()
        query = SupportQuery(text="How do I configure ACL users in Redis 5?", redis_version="5.0")
        route, _ = router.route(query)
        unrelated = RedisDocument(
            source_id="eviction",
            title="Memory pressure",
            text="LRU removes old keys when memory is full.",
            source_url="https://redis.io/docs/eviction",
            module="configuration",
        )
        chunks = LexicalRetriever([unrelated]).retrieve(query, route)
        self.assertEqual(chunks, [])

    def test_retriever_ignores_generic_redis_and_version_tokens(self) -> None:
        router = RuleBasedRouter()
        query = SupportQuery(
            text="How do I enable quantum index acceleration in Redis 7?",
            redis_version="7.2",
        )
        route, _ = router.route(query)
        unrelated = RedisDocument(
            source_id="generic",
            title="Redis 7 persistence",
            text="Redis 7 stores durable data with AOF.",
            source_url="https://redis.io/docs/persistence",
            module="persistence",
        )
        chunks = LexicalRetriever([unrelated]).retrieve(query, route)
        self.assertEqual(chunks, [])

    def test_pipeline_abstains_on_invalid_citation(self) -> None:
        pipeline = RedisDoctorPipeline(
            RuleBasedRouter(),
            LexicalRetriever(self.documents),
            generator=InvalidCitationGenerator(),
            minimum_score=0.1,
        )
        answer = pipeline.answer(SupportQuery(text="AOF persistence", redis_version="7.2"))
        self.assertTrue(answer.abstained)
        self.assertEqual(answer.reason, "invalid_citations")
        self.assertIn("拒绝", answer.answer)

    def test_pipeline_abstains_without_evidence(self) -> None:
        pipeline = RedisDoctorPipeline(RuleBasedRouter(), LexicalRetriever(self.documents))
        answer = pipeline.answer(SupportQuery(text="unrelated frobnicator setting"))
        self.assertTrue(answer.abstained)
        self.assertEqual(answer.reason, "insufficient_evidence")
        self.assertIn("证据", answer.answer)

    def test_pipeline_returns_structured_refusal_when_generator_fails(self) -> None:
        pipeline = RedisDoctorPipeline(
            RuleBasedRouter(),
            LexicalRetriever(self.documents),
            generator=FailingGenerator(),
            minimum_score=0.1,
        )
        answer = pipeline.answer(SupportQuery(text="AOF persistence", redis_version="7.2"))
        self.assertTrue(answer.abstained)
        self.assertEqual(answer.reason, "generation_error")
        self.assertEqual(answer.error_code, "timeout")
        self.assertIn("暂时无法", answer.answer)


if __name__ == "__main__":
    unittest.main()
