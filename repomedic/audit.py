from __future__ import annotations

from datetime import UTC, datetime
from threading import Lock
from typing import Any, Literal

from pydantic import BaseModel, Field


class AuditEvent(BaseModel):
    sequence: int = Field(ge=1)
    timestamp: datetime
    phase: str
    kind: Literal["model_response", "tool_call", "tool_result", "error", "decision"]
    round: int | None = Field(default=None, ge=0)
    success: bool = True
    token_usage: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0, ge=0)
    payload: dict[str, Any] = Field(default_factory=dict)


class AuditSnapshot(BaseModel):
    prompt_version: str
    events: list[AuditEvent]
    attempted_tool_calls: int = Field(ge=0)
    invalid_tool_calls: int = Field(ge=0)
    token_usage: int = Field(ge=0)

    @property
    def invalid_tool_rate(self) -> float:
        if not self.attempted_tool_calls:
            return 0.0
        return self.invalid_tool_calls / self.attempted_tool_calls


class AuditTrail:
    """线程安全地保存模型可见输出和工具调用，不包含隐藏思维链。"""

    def __init__(self, prompt_version: str) -> None:
        self.prompt_version = prompt_version
        self._events: list[AuditEvent] = []
        self._lock = Lock()

    def record(
        self,
        *,
        phase: str,
        kind: Literal["model_response", "tool_call", "tool_result", "error", "decision"],
        payload: dict[str, Any] | None = None,
        round: int | None = None,
        success: bool = True,
        token_usage: int = 0,
        duration_ms: float = 0,
    ) -> None:
        with self._lock:
            self._events.append(
                AuditEvent(
                    sequence=len(self._events) + 1,
                    timestamp=datetime.now(UTC),
                    phase=phase,
                    kind=kind,
                    round=round,
                    success=success,
                    token_usage=token_usage,
                    duration_ms=duration_ms,
                    payload=payload or {},
                )
            )

    def snapshot(self) -> AuditSnapshot:
        with self._lock:
            events = [event.model_copy(deep=True) for event in self._events]
        attempted = sum(event.kind == "tool_call" for event in events)
        invalid = sum(event.kind == "tool_result" and not event.success for event in events)
        return AuditSnapshot(
            prompt_version=self.prompt_version,
            events=events,
            attempted_tool_calls=attempted,
            invalid_tool_calls=invalid,
            token_usage=sum(
                event.token_usage for event in events if event.kind == "model_response"
            ),
        )
