from __future__ import annotations

import io
import json
import unittest
import urllib.error
from typing import Self
from unittest.mock import patch

from redisdoctor.generation import OpenAICompatibleGenerator
from redisdoctor.models import RetrievedChunk, SupportQuery
from redisdoctor.pipeline import GenerationError


def context() -> RetrievedChunk:
    return RetrievedChunk(
        source_id="aof-7",
        title="AOF",
        text="AOF rewrite compacts the log.",
        source_url="https://redis.io/docs/aof",
        redis_version="7.2",
        module="persistence",
        score=2,
    )


class FakeResponse(io.BytesIO):
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class RedisDoctorGenerationTest(unittest.TestCase):
    @patch("redisdoctor.generation.urllib.request.urlopen")
    def test_parses_fenced_structured_response(self, urlopen: object) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "```json\n{\"answer\":\"证据回答\",\"cited_source_ids\":[\"aof-7\"]}\n```"
                    }
                }
            ],
            "usage": {"total_tokens": 42},
        }
        urlopen.return_value = FakeResponse(json.dumps(payload).encode())  # type: ignore[attr-defined]
        generator = OpenAICompatibleGenerator("http://localhost:8080/v1", "local", "")
        result = generator.generate(SupportQuery(text="AOF是什么？"), [context()])
        self.assertEqual(result.answer, "证据回答")
        self.assertEqual(result.cited_source_ids, ["aof-7"])
        self.assertEqual(result.token_usage, 42)

    @patch("redisdoctor.generation.urllib.request.urlopen")
    def test_wraps_timeout_as_generation_error(self, urlopen: object) -> None:
        urlopen.side_effect = TimeoutError("slow")  # type: ignore[attr-defined]
        generator = OpenAICompatibleGenerator("http://localhost:8080/v1", "local", "")
        with self.assertRaisesRegex(GenerationError, "timeout"):
            generator.generate(SupportQuery(text="AOF是什么？"), [context()])

    @patch("redisdoctor.generation.urllib.request.urlopen")
    def test_wraps_http_error_as_generation_error(self, urlopen: object) -> None:
        urlopen.side_effect = urllib.error.HTTPError(  # type: ignore[attr-defined]
            "http://localhost", 500, "error", {}, None
        )
        generator = OpenAICompatibleGenerator("http://localhost:8080/v1", "local", "")
        with self.assertRaisesRegex(GenerationError, "http_error"):
            generator.generate(SupportQuery(text="AOF是什么？"), [context()])

    @patch("redisdoctor.generation.urllib.request.urlopen")
    def test_rejects_malformed_model_json(self, urlopen: object) -> None:
        payload = {"choices": [{"message": {"content": "not-json"}}]}
        urlopen.return_value = FakeResponse(json.dumps(payload).encode())  # type: ignore[attr-defined]
        generator = OpenAICompatibleGenerator("http://localhost:8080/v1", "local", "")
        with self.assertRaisesRegex(GenerationError, "invalid_json"):
            generator.generate(SupportQuery(text="AOF是什么？"), [context()])
