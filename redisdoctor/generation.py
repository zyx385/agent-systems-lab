from __future__ import annotations

import json
import urllib.error
import urllib.request
from json import JSONDecodeError

from pydantic import ValidationError

from .models import RetrievedChunk, SupportQuery
from .pipeline import GeneratedResponse, GenerationError


def _extract_json_object(content: str) -> object:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1]).strip()
    start = stripped.find("{")
    if start < 0:
        raise JSONDecodeError("没有找到JSON对象", stripped, 0)
    value, _ = json.JSONDecoder().raw_decode(stripped[start:])
    return value


class OpenAICompatibleGenerator:
    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str,
        timeout: int = 60,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def generate(
        self, query: SupportQuery, contexts: list[RetrievedChunk]
    ) -> GeneratedResponse:
        schema = GeneratedResponse.model_json_schema()
        evidence = [
            {
                "source_id": context.source_id,
                "title": context.title,
                "text": context.text,
                "source_url": context.source_url,
                "redis_version": context.redis_version,
            }
            for context in contexts
        ]
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是Redis技术支持助手。只能依据给定证据回答；不得引用证据列表外的"
                        "source_id。证据不足时answer写明无法确认，且cited_source_ids为空。"
                        "只返回符合Schema的JSON："
                        + json.dumps(schema, ensure_ascii=False)
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"query": query.model_dump(mode="json"), "evidence": evidence},
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.endpoint}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
            content = result["choices"][0]["message"]["content"]
            generated = GeneratedResponse.model_validate(_extract_json_object(content))
            generated.token_usage = int(result.get("usage", {}).get("total_tokens", 0))
            return generated
        except (TimeoutError, urllib.error.URLError) as exc:
            if isinstance(exc, urllib.error.HTTPError):
                raise GenerationError("http_error", f"HTTP {exc.code}") from exc
            code = "timeout" if isinstance(exc, TimeoutError) else "connection_error"
            raise GenerationError(code, str(exc)) from exc
        except (KeyError, TypeError, ValueError, JSONDecodeError, ValidationError) as exc:
            raise GenerationError("invalid_json", str(exc)) from exc
