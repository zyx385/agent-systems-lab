from __future__ import annotations

import json
import urllib.request
from typing import Protocol

from .models import RouteDecision, SupportQuery


class QueryRouter(Protocol):
    def route(self, query: SupportQuery) -> tuple[RouteDecision, int]: ...


class RuleBasedRouter:
    _INTENTS = {
        "persistence": {"rdb", "aof", "appendonly", "snapshot", "persistence", "持久化"},
        "replication": {"replica", "replication", "sentinel", "cluster", "主从", "哨兵"},
        "performance": {"slow", "latency", "memory", "cpu", "performance", "延迟", "性能"},
        "configuration": {"config", "配置", "maxmemory", "timeout"},
        "command": {"command", "get", "set", "scan", "keys", "命令"},
    }

    def route(self, query: SupportQuery) -> tuple[RouteDecision, int]:
        lowered = query.text.lower()
        scores = {
            intent: sum(keyword in lowered for keyword in keywords)
            for intent, keywords in self._INTENTS.items()
        }
        intent, score = max(scores.items(), key=lambda item: item[1])
        if score == 0:
            intent = "unknown"
        mode = "global" if any(word in lowered for word in ("compare", "architecture", "区别")) else "local"
        if intent in {"persistence", "replication"} and mode == "local":
            mode = "hybrid"
        version_suffix = f" Redis {query.redis_version}" if query.redis_version else ""
        return (
            RouteDecision(
                intent=intent,  # type: ignore[arg-type]
                module=intent if intent != "unknown" else "general",
                rewritten_query=f"{query.text.strip()}{version_suffix}",
                mode=mode,
                confidence=0.8 if score else 0.25,
            ),
            0,
        )


class OpenAICompatibleRouter:
    """通过OpenAI兼容API调用基座模型或QLoRA模型的适配器。"""

    def __init__(self, endpoint: str, model: str, api_key: str, timeout: int = 30) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def route(self, query: SupportQuery) -> tuple[RouteDecision, int]:
        schema = RouteDecision.model_json_schema()
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "对Redis技术支持请求进行路由。只返回符合以下Schema的JSON："
                        + json.dumps(schema, ensure_ascii=False)
                    ),
                },
                {"role": "user", "content": query.model_dump_json()},
            ],
        }
        request = urllib.request.Request(
            f"{self.endpoint}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        content = result["choices"][0]["message"]["content"]
        tokens = int(result.get("usage", {}).get("total_tokens", 0))
        return RouteDecision.model_validate_json(content), tokens
