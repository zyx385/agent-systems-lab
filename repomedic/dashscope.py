from __future__ import annotations

import http.client
import json
import os
import re
import shlex
import time
import urllib.error
import urllib.request
from pathlib import Path
from threading import Lock
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, Field, ValidationError

from .audit import AuditTrail
from .models import (
    CoordinatorRoute,
    DiagnosisInput,
    DiagnosisPlan,
    Evidence,
    Hypothesis,
    RoleFinding,
    SharedCaseState,
    SpecialistRole,
    Verification,
)
from .tools import CommandResult, RepositoryTools

PROMPT_VERSION = "repomedic-dashscope-v3"
StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)
SHARED_CONSTRAINTS = (
    "必须依据真实工具结果；文件路径必须来自仓库证据；测试只能通过run_command执行；"
    "run_command只能使用python -m pytest、pytest或python -m unittest开头的命令，"
    "禁止python -c、Shell运算符、timeout命令和未声明的pytest插件参数；"
    "没有充分证据或没有实际验证时必须拒绝确认；最终输出必须是指定JSON结构。"
)


def _request_context(request: DiagnosisInput) -> str:
    """不把临时仓库绝对路径泄露给模型，避免案例目录名污染检索词。"""
    return json.dumps(
        {"issue": request.issue, "traceback": request.traceback}, ensure_ascii=False
    )


class ModelAPIError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class AgentRunError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ChatCompletionClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], int]: ...


class DashScopeClient:
    """调用百炼OpenAI兼容Chat Completions接口。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int = 60,
        max_retries: int = 2,
        retry_delay: float = 1.0,
        max_output_tokens: int = 2048,
    ) -> None:
        if not api_key or not base_url or not model:
            raise ValueError("百炼API Key、Base URL和模型名称不能为空")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_output_tokens = max_output_tokens

    @classmethod
    def from_env(cls) -> DashScopeClient:
        missing = [
            name
            for name in ("DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL", "DASHSCOPE_MODEL")
            if not os.getenv(name)
        ]
        if missing:
            raise RuntimeError(f"缺少百炼环境变量：{', '.join(missing)}")
        return cls(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url=os.environ["DASHSCOPE_BASE_URL"],
            model=os.environ["DASHSCOPE_MODEL"],
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], int]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "enable_thinking": False,
            "max_tokens": self.max_output_tokens,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        result: dict[str, Any] | None = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.load(response)
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
                if exc.code not in {429, 500, 502, 503, 504} or attempt == self.max_retries:
                    code = _http_error_code(exc.code, detail)
                    raise ModelAPIError(code, f"百炼API返回HTTP {exc.code}：{detail}") from exc
            except urllib.error.URLError as exc:
                if attempt == self.max_retries:
                    code = "api_timeout" if isinstance(exc.reason, TimeoutError) else "connection_error"
                    raise ModelAPIError(code, f"无法连接百炼API：{exc.reason}") from exc
            except TimeoutError as exc:
                if attempt == self.max_retries:
                    raise ModelAPIError("api_timeout", "百炼API请求超时") from exc
            except http.client.RemoteDisconnected as exc:
                if attempt == self.max_retries:
                    raise ModelAPIError("connection_error", "百炼API提前关闭连接") from exc
            time.sleep(self.retry_delay * (2**attempt))
        if result is None:
            raise ModelAPIError("empty_api_result", "百炼API没有返回结果")
        try:
            message = result["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelAPIError("invalid_api_response", "百炼API响应缺少choices[0].message") from exc
        tokens = int(result.get("usage", {}).get("total_tokens", 0))
        return message, tokens


class InvestigationHypothesis(BaseModel):
    summary: str
    root_file: str
    confidence: float = Field(ge=0, le=1)
    suggested_patch: str = ""


class InvestigationOutput(BaseModel):
    hypotheses: list[InvestigationHypothesis] = Field(default_factory=list)


class TriageOutput(BaseModel):
    search_terms: list[str]
    test_commands: list[list[str] | str] = Field(default_factory=list)
    rationale: str


class VerificationDecision(BaseModel):
    hypothesis_index: int = Field(ge=0)
    confirmed: bool
    notes: str


class VerificationOutput(BaseModel):
    decisions: list[VerificationDecision] = Field(default_factory=list)


class SingleAgentOutput(BaseModel):
    hypotheses: list[InvestigationHypothesis] = Field(default_factory=list)
    decisions: list[VerificationDecision] = Field(default_factory=list)


class RoleFindingOutput(BaseModel):
    question: str
    evidence_paths: list[str] = Field(min_length=1)
    candidate_files: list[str] = Field(min_length=1)
    primary_file: str = Field(min_length=3)
    claim: str
    confidence: float = Field(ge=0, le=1)
    open_questions: list[str] = Field(default_factory=list)


class DashScopeBackend:
    """使用百炼模型完成分诊、工具调查和独立验证。"""

    def __init__(
        self,
        client: ChatCompletionClient,
        max_tool_rounds: int = 4,
        *,
        audit: AuditTrail | None = None,
        max_total_tokens: int = 50_000,
    ) -> None:
        self.client = client
        self.max_tool_rounds = max_tool_rounds
        self.audit = audit
        self.max_total_tokens = max_total_tokens
        self._used_tokens = 0
        self._budget_lock = Lock()

    @classmethod
    def from_env(cls, *, audit: AuditTrail | None = None) -> DashScopeBackend:
        return cls(DashScopeClient.from_env(), audit=audit)

    def triage(self, request: DiagnosisInput) -> tuple[DiagnosisPlan, int]:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是代码故障分诊员。只输出JSON，不要输出Markdown。"
                    "生成1到3个精确源码检索词；测试命令只能使用pytest或unittest。"
                    f"共同约束：{SHARED_CONSTRAINTS}"
                    f"输出必须符合Schema：{json.dumps(DiagnosisPlan.model_json_schema(), ensure_ascii=False)}"
                ),
            },
            {"role": "user", "content": _request_context(request)},
        ]
        message, tokens = self._complete(messages, phase="triage", round=0)
        raw_plan = self._structured_output(message, TriageOutput, phase="triage").model_dump()
        commands = raw_plan.get("test_commands", [])
        raw_plan["test_commands"] = [
            shlex.split(command) if isinstance(command, str) else command
            for command in commands
        ]
        plan = DiagnosisPlan.model_validate(raw_plan)
        self._record_decision("triage", {"plan": plan.model_dump(mode="json")})
        return plan, tokens

    def route_complex(self, request: DiagnosisInput) -> tuple[CoordinatorRoute, int]:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是复杂度路由协调员。只输出JSON，不得确认根因。"
                    "简单、单文件且证据一致的问题选simple和空roles；跨模块、配置、状态、"
                    "权限或异步问题选complex，并选择2到3个职责不同的专项角色。"
                    f"输出必须符合Schema：{json.dumps(CoordinatorRoute.model_json_schema(), ensure_ascii=False)}"
                ),
            },
            {"role": "user", "content": _request_context(request)},
        ]
        message, tokens = self._complete(messages, phase="coordinator", round=0)
        route = self._structured_output(message, CoordinatorRoute, phase="coordinator")
        self._record_decision("coordinator", {"route": route.model_dump(mode="json")})
        return route, tokens

    def investigate_role(
        self,
        request: DiagnosisInput,
        route: CoordinatorRoute,
        state: SharedCaseState,
        role: SpecialistRole,
        tools: RepositoryTools,
    ) -> tuple[RoleFinding, list[Evidence], int]:
        question = next(
            (item for item in route.evidence_questions if item),
            "What evidence supports the diagnosis?",
        )
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    f"你是专项代码调查员，职责是{role}。只回答职责范围内的问题。"
                    "必须依据工具结果；search_repository只接受字面量源码标识符，"
                    "禁止传正则表达式、案例ID或目录名；不要重新读取共享证据中已有的完整文件。"
                    f"共同约束：{SHARED_CONSTRAINTS}"
                    "最终只输出JSON：question、evidence_paths、candidate_files、primary_file、claim、"
                    "confidence、open_questions。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "issue": request.issue,
                        "traceback": request.traceback,
                        "route": route.model_dump(),
                        "role_question": question,
                        "shared_evidence": [item.model_dump() for item in state.evidence],
                        "already_read_files": sorted(state.full_file_reads),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        final, evidence, _commands, tokens = self._tool_loop(
            messages,
            _investigation_tools(),
            tools,
            phase=f"role:{role}",
            first_tool_choice={"type": "function", "function": {"name": "search_repository"}},
        )
        output = self._structured_output(final, RoleFindingOutput, phase=f"role:{role}")
        accepted = state.add_evidence(evidence, role)
        path_to_ids: dict[str, list[str]] = {}
        for item in accepted:
            path_to_ids.setdefault(item.path, []).append(item.id)
        evidence_ids = [
            evidence_id
            for path in output.evidence_paths
            for evidence_id in path_to_ids.get(path, [])
        ]
        if not evidence_ids:
            raise AgentRunError("schema_validation_error", "角色结论没有引用本轮真实证据")
        finding = RoleFinding(
            role=role,
            question=output.question,
            evidence_ids=list(dict.fromkeys(evidence_ids)),
            candidate_files=output.candidate_files,
            primary_file=output.primary_file,
            claim=output.claim,
            confidence=output.confidence,
            open_questions=output.open_questions,
        )
        state.findings.append(finding)
        self._record_decision(f"role:{role}", {"finding": finding.model_dump(mode="json")})
        return finding, accepted, tokens

    def investigate(
        self,
        request: DiagnosisInput,
        plan: DiagnosisPlan,
        search_term: str,
        tools: RepositoryTools,
    ) -> tuple[list[Evidence], list[Hypothesis], int]:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "你是代码仓库调查员。必须先调用工具检查真实文件，不能凭空猜测。"
                    f"共同约束：{SHARED_CONSTRAINTS}"
                    "最终只输出JSON：{\"hypotheses\":[{\"summary\":字符串,"
                    "\"root_file\":仓库相对路径,\"confidence\":0到1,"
                    "\"suggested_patch\":字符串}]}。root_file必须来自工具结果。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "issue": request.issue,
                        "traceback": request.traceback,
                        "plan": plan.model_dump(),
                        "current_search_term": search_term,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        first_choice = {
            "type": "function",
            "function": {"name": "search_repository"},
        }
        final, evidence, _commands, tokens = self._tool_loop(
            messages,
            _investigation_tools(),
            tools,
            phase=f"investigate:{search_term}",
            first_tool_choice=first_choice,
        )
        output = self._structured_output(final, InvestigationOutput, phase=f"investigate:{search_term}")
        hypotheses: list[Hypothesis] = []
        for candidate in output.hypotheses:
            evidence_ids = [item.id for item in evidence if item.path == candidate.root_file]
            if not evidence_ids:
                continue
            hypotheses.append(
                Hypothesis(
                    summary=candidate.summary,
                    root_file=candidate.root_file,
                    confidence=candidate.confidence,
                    evidence_ids=evidence_ids,
                    suggested_patch=candidate.suggested_patch,
                )
            )
        self._record_decision(
            f"investigate:{search_term}",
            {"hypotheses": [item.model_dump(mode="json") for item in hypotheses]},
        )
        return evidence, hypotheses, tokens

    def verify(
        self,
        request: DiagnosisInput,
        hypotheses: list[Hypothesis],
        evidence: list[Evidence],
        tools: RepositoryTools,
    ) -> tuple[list[Verification], int]:
        if not hypotheses:
            return [], 0
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "你是独立验证员。优先调用run_command运行针对性测试，也可以read_file复核源码。"
                    f"共同约束：{SHARED_CONSTRAINTS}"
                    "没有实际执行验证命令时confirmed必须为false。最终只输出JSON："
                    "{\"decisions\":[{\"hypothesis_index\":整数,\"confirmed\":布尔值,"
                    "\"notes\":字符串}]}。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "issue": request.issue,
                        "traceback": request.traceback,
                        "hypotheses": [item.model_dump() for item in hypotheses],
                        "evidence": [item.model_dump() for item in evidence],
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        first_choice = {
            "type": "function",
            "function": {"name": "run_command"},
        }
        final, _evidence, executions, tokens = self._tool_loop(
            messages,
            _verification_tools(),
            tools,
            phase="verify",
            first_tool_choice=first_choice,
        )
        output = self._structured_output(final, VerificationOutput, phase="verify")
        verifications: list[Verification] = []
        for decision in output.decisions:
            if decision.hypothesis_index >= len(hypotheses):
                continue
            verifications.append(
                Verification(
                    hypothesis_index=decision.hypothesis_index,
                    confirmed=decision.confirmed
                    and any(
                        _test_failure_supports_hypothesis(result, hypotheses[decision.hypothesis_index], tools)
                        for result in executions
                    ),
                    commands=[result.command for result in executions],
                    notes=(
                        f"{decision.notes} 实际命令返回码："
                        f"{[result.returncode for result in executions]}。"
                    ),
                )
            )
        self._record_decision(
            "verify",
            {"verifications": [item.model_dump(mode="json") for item in verifications]},
        )
        return verifications, tokens

    def diagnose_single_agent(
        self, request: DiagnosisInput, tools: RepositoryTools
    ) -> tuple[list[Evidence], list[Hypothesis], list[Verification], int]:
        """用一个提示词和一个工具循环完成检索、判断与验证。"""
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "你是单一代码故障诊断Agent。必须先搜索真实仓库，再读取相关源码和测试，"
                    f"共同约束：{SHARED_CONSTRAINTS}"
                    "并调用run_command执行针对性测试；不能把源码阅读当成运行验证。最终只输出JSON："
                    "{\"hypotheses\":[{\"summary\":字符串,\"root_file\":仓库相对路径,"
                    "\"confidence\":0到1,\"suggested_patch\":字符串}],"
                    "\"decisions\":[{\"hypothesis_index\":整数,\"confirmed\":布尔值,"
                    "\"notes\":字符串}]}。root_file必须来自工具结果。"
                ),
            },
            {"role": "user", "content": _request_context(request)},
        ]
        first_choice = {
            "type": "function",
            "function": {"name": "search_repository"},
        }
        final, evidence, executions, tokens = self._tool_loop(
            messages,
            _single_agent_tools(),
            tools,
            phase="single",
            first_tool_choice=first_choice,
        )
        output = self._structured_output(final, SingleAgentOutput, phase="single")
        hypotheses: list[Hypothesis] = []
        for candidate in output.hypotheses:
            evidence_ids = [item.id for item in evidence if item.path == candidate.root_file]
            if not evidence_ids:
                continue
            hypotheses.append(
                Hypothesis(
                    summary=candidate.summary,
                    root_file=candidate.root_file,
                    confidence=candidate.confidence,
                    evidence_ids=evidence_ids,
                    suggested_patch=candidate.suggested_patch,
                )
            )
        verifications: list[Verification] = []
        for decision in output.decisions:
            if decision.hypothesis_index >= len(hypotheses):
                continue
            verifications.append(
                Verification(
                    hypothesis_index=decision.hypothesis_index,
                    confirmed=decision.confirmed
                    and any(
                        _test_failure_supports_hypothesis(result, hypotheses[decision.hypothesis_index], tools)
                        for result in executions
                    ),
                    commands=[result.command for result in executions],
                    notes=(
                        f"{decision.notes} 实际命令返回码："
                        f"{[result.returncode for result in executions]}。"
                    ),
                )
            )
        self._record_decision(
            "single",
            {
                "hypotheses": [item.model_dump(mode="json") for item in hypotheses],
                "verifications": [item.model_dump(mode="json") for item in verifications],
            },
        )
        return evidence, hypotheses, verifications, tokens

    def _tool_loop(
        self,
        messages: list[dict[str, Any]],
        definitions: list[dict[str, Any]],
        repository_tools: RepositoryTools,
        *,
        phase: str,
        first_tool_choice: str | dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], list[Evidence], list[CommandResult], int]:
        evidence: list[Evidence] = []
        executions: list[CommandResult] = []
        token_usage = 0
        choice = first_tool_choice
        for round_index in range(self.max_tool_rounds):
            message, tokens = self._complete(
                messages,
                phase=phase,
                round=round_index,
                tools=definitions,
                tool_choice=choice,
            )
            token_usage += tokens
            choice = None
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                return message, evidence, executions, token_usage
            messages.append(message)
            for call in tool_calls:
                function = call.get("function", {})
                name = str(function.get("name") or "")
                raw_arguments = str(function.get("arguments") or "{}")
                try:
                    parsed_arguments: Any = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    parsed_arguments = None
                if self.audit:
                    self.audit.record(
                        phase=phase,
                        kind="tool_call",
                        round=round_index,
                        payload={
                            "call_id": str(call.get("id") or ""),
                            "name": name,
                            "arguments_raw": raw_arguments,
                            "arguments": parsed_arguments,
                        },
                    )
                tool_started = time.perf_counter()
                result, found, execution = _execute_tool(call, repository_tools)
                try:
                    decoded_result: Any = json.loads(result)
                except json.JSONDecodeError:
                    decoded_result = result
                success = not (
                    isinstance(decoded_result, dict) and "error" in decoded_result
                )
                if self.audit:
                    self.audit.record(
                        phase=phase,
                        kind="tool_result",
                        round=round_index,
                        success=success,
                        duration_ms=(time.perf_counter() - tool_started) * 1000,
                        payload={
                            "call_id": str(call.get("id") or ""),
                            "name": name,
                            "result": decoded_result,
                        },
                    )
                evidence.extend(found)
                if execution:
                    executions.append(execution)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(call.get("id") or ""),
                        "content": result,
                    }
                )
        messages.append(
            {
                "role": "user",
                "content": "工具调用轮次已用完。请根据已有工具结果立即输出最终JSON。",
            }
        )
        message, tokens = self._complete(
            messages,
            phase=phase,
            round=self.max_tool_rounds,
        )
        return message, evidence, executions, token_usage + tokens

    def _structured_output(
        self,
        message: dict[str, Any],
        schema: type[StructuredOutput],
        *,
        phase: str,
    ) -> StructuredOutput:
        try:
            return schema.model_validate_json(_json_content(message))
        except (AgentRunError, ValidationError) as original_error:
            raw_content = message.get("content")
            repair_messages = [
                {
                    "role": "system",
                    "content": (
                        "你是JSON格式修复器。不得调用工具、不得新增证据或结论；"
                        "只将给定原始输出重排为符合Schema的单个JSON对象。"
                        f"Schema：{json.dumps(schema.model_json_schema(), ensure_ascii=False)}"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"invalid_output": raw_content, "error": str(original_error)},
                        ensure_ascii=False,
                    ),
                },
            ]
            repaired, _tokens = self._complete(
                repair_messages,
                phase=f"{phase}:json_repair",
                round=self.max_tool_rounds + 1,
            )
            return schema.model_validate_json(_json_content(repaired))

    def _complete(
        self,
        messages: list[dict[str, Any]],
        *,
        phase: str,
        round: int,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], int]:
        started = time.perf_counter()
        try:
            message, tokens = self.client.complete(
                messages,
                tools=tools,
                tool_choice=tool_choice,
            )
        except Exception as exc:
            if self.audit:
                self.audit.record(
                    phase=phase,
                    kind="error",
                    round=round,
                    success=False,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    payload={
                        "error_type": getattr(exc, "code", type(exc).__name__),
                        "message": str(exc),
                    },
                )
            raise
        duration_ms = (time.perf_counter() - started) * 1000
        if self.audit:
            self.audit.record(
                phase=phase,
                kind="model_response",
                round=round,
                token_usage=tokens,
                duration_ms=duration_ms,
                payload={
                    "message": message,
                    "requested_tool_choice": tool_choice,
                },
            )
        with self._budget_lock:
            self._used_tokens += tokens
            exceeded = self._used_tokens > self.max_total_tokens
        if exceeded:
            error = AgentRunError(
                "token_limit",
                f"本次诊断Token累计{self._used_tokens}，超过上限{self.max_total_tokens}",
            )
            if self.audit:
                self.audit.record(
                    phase=phase,
                    kind="error",
                    round=round,
                    success=False,
                    payload={"error_type": error.code, "message": str(error)},
                )
            raise error
        return message, tokens

    def _record_decision(self, phase: str, payload: dict[str, Any]) -> None:
        if self.audit:
            self.audit.record(phase=phase, kind="decision", payload=payload)


def _json_content(message: dict[str, Any]) -> str:
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise AgentRunError("invalid_json", "模型没有返回可解析的JSON内容")
    stripped = content.strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        stripped = stripped[7:-3].strip()
    elif stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped[3:-3].strip()
    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, character in enumerate(stripped):
            if character not in "{[":
                continue
            try:
                value, end = decoder.raw_decode(stripped[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return stripped[index : index + end]
    raise AgentRunError("invalid_json", "模型回复中没有可解析的JSON对象")


def _http_error_code(status: int, detail: str) -> str:
    lowered = detail.lower()
    if "insufficient_quota" in lowered or "free quota exhausted" in lowered:
        return "insufficient_quota"
    if status in {408, 504}:
        return "api_timeout"
    if status == 429:
        return "rate_limit"
    if status in {401, 403}:
        return "authentication_error"
    if status == 400 and any(
        marker in lowered for marker in ("context_length", "maximum context", "token limit")
    ):
        return "token_limit"
    if status >= 500:
        return "provider_error"
    return "api_error"


def _test_failure_supports_hypothesis(
    result: CommandResult, hypothesis: Hypothesis, tools: RepositoryTools
) -> bool:
    if result.timed_out:
        return False
    command = result.command
    is_pytest = command[:3] == ["python", "-m", "pytest"] or command[:1] == ["pytest"]
    is_unittest = command[:3] == ["python", "-m", "unittest"]
    if is_pytest and result.returncode != 1:
        return False
    if is_unittest and result.returncode == 0:
        return False
    if not is_pytest and not is_unittest:
        return False
    output = f"{result.stdout}\n{result.stderr}".replace("\\", "/")
    normalized_root = hypothesis.root_file.replace("\\", "/")
    if normalized_root in output or Path(normalized_root).name in output:
        return True
    module_name = normalized_root.removesuffix(".py").replace("/", ".")
    test_paths = {
        match.replace("\\", "/")
        for match in re.findall(r"(?:tests?|test)/[^\s:'\"]+\.py", output)
    }
    test_paths.update(
        part.replace("\\", "/").split("::", 1)[0]
        for part in command
        if ".py" in part and part.startswith(("test", "tests"))
    )
    for test_path in test_paths:
        try:
            content = tools.read(test_path, start=1, end=500)
        except (OSError, ValueError):
            continue
        source = re.sub(r"^\d+:\s?", "", content, flags=re.M)
        if f"from {module_name} import" in source or f"import {module_name}" in source:
            return True
        if _imports_reach_module(source, module_name, tools):
            return True
    return False


def _imports_reach_module(content: str, target_module: str, tools: RepositoryTools) -> bool:
    """确认失败测试能经由本地 Python 导入链到达候选文件。"""
    pending = _imported_local_modules(content)
    visited: set[str] = set()
    while pending and len(visited) < 32:
        module = pending.pop()
        if module == target_module:
            return True
        if module in visited:
            continue
        visited.add(module)
        path = f"{module.replace('.', '/')}.py"
        try:
            imported_content = tools.read(path, start=1, end=500)
        except (OSError, ValueError):
            continue
        pending.update(_imported_local_modules(imported_content) - visited)
    return False


def _imported_local_modules(content: str) -> set[str]:
    source = re.sub(r"^\d+:\s?", "", content, flags=re.M)
    modules = set(re.findall(r"^\s*from\s+(app(?:\.[A-Za-z_]\w*)*)\s+import\s+", source, re.M))
    modules.update(re.findall(r"^\s*import\s+(app(?:\.[A-Za-z_]\w*)*)", source, re.M))
    return modules


def _execute_tool(
    call: dict[str, Any], tools: RepositoryTools
) -> tuple[str, list[Evidence], CommandResult | None]:
    function = call.get("function", {})
    name = function.get("name")
    raw_arguments = function.get("arguments") or "{}"
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"工具参数不是有效JSON：{exc}"}, ensure_ascii=False), [], None
    try:
        if name == "search_repository":
            rows = tools.search(str(arguments["pattern"]), limit=int(arguments.get("limit", 8)))
            return json.dumps([row.model_dump() for row in rows], ensure_ascii=False), rows, None
        if name == "read_file":
            path = str(arguments["path"])
            start = int(arguments.get("start", 1))
            end = int(arguments.get("end", start + 199))
            content = tools.read(path, start=start, end=end)
            item = Evidence(
                id=f"read-{len(content)}-{start}",
                kind="source",
                path=path,
                line=start,
                excerpt=content[:1000],
            )
            return json.dumps({"content": content}, ensure_ascii=False), [item], None
        if name == "call_graph":
            graph = tools.call_graph(str(arguments["path"]))
            item = Evidence(
                id=f"graph-{len(graph.definitions)}",
                kind="static",
                path=graph.path,
                excerpt=graph.model_dump_json()[:1000],
            )
            return graph.model_dump_json(), [item], None
        if name == "run_command":
            command = [str(part) for part in arguments["command"]]
            result = tools.run(command)
            return result.model_dump_json(), [], result
        return json.dumps({"error": f"未知工具：{name}"}, ensure_ascii=False), [], None
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False), [], None


def _investigation_tools() -> list[dict[str, Any]]:
    return [
        _tool_definition(
            "search_repository",
            "在仓库Python源码中按字面量搜索，返回文件、行号和代码片段。",
            {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 8},
                },
                "required": ["pattern"],
            },
        ),
        _tool_definition(
            "read_file",
            "读取仓库内文件的指定行范围。",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start": {"type": "integer", "minimum": 1},
                    "end": {"type": "integer", "minimum": 1},
                },
                "required": ["path"],
            },
        ),
        _tool_definition(
            "call_graph",
            "解析Python文件并返回函数到调用名称的静态关系。",
            {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        ),
    ]


def _verification_tools() -> list[dict[str, Any]]:
    return [
        _investigation_tools()[1],
        _tool_definition(
            "run_command",
            "运行受白名单和超时保护的pytest、unittest或静态检查命令。",
            {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    }
                },
                "required": ["command"],
            },
        ),
    ]


def _single_agent_tools() -> list[dict[str, Any]]:
    return [*_investigation_tools(), _verification_tools()[1]]


def _tool_definition(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }
