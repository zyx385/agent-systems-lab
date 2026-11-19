from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from typing import Any
from unittest.mock import patch

from repomedic.audit import AuditTrail
from repomedic.dashscope import (
    PROMPT_VERSION,
    AgentRunError,
    DashScopeBackend,
    DashScopeClient,
    ModelAPIError,
    _http_error_code,
    _json_content,
    _request_context,
    _test_failure_supports_hypothesis,
)
from repomedic.models import DiagnosisInput, DiagnosisPlan, Evidence, Hypothesis
from repomedic.tools import CommandResult, RepositoryTools


class FakeClient:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.responses = list(messages)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], int]:
        self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
        return self.responses.pop(0), 10


class FakeRepositoryTools(RepositoryTools):
    def run(self, command: list[str]) -> CommandResult:
        self.call_count += 1
        return CommandResult(
            command=command,
            returncode=1,
            stdout="FAILED tests/test_pricing.py app/pricing.py",
            stderr="1 failed",
        )


class DashScopeBackendTest(unittest.TestCase):
    def test_request_context_omits_repository_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = DiagnosisInput(repository=Path(directory), issue="cache key is wrong")
            context = _request_context(request)
        self.assertNotIn(directory, context)
        self.assertIn("cache key is wrong", context)

    def test_quota_error_is_not_mislabeled_as_authentication(self) -> None:
        code = _http_error_code(403, '{"code":"insufficient_quota"}')
        self.assertEqual(code, "insufficient_quota")

    def test_pytest_usage_error_does_not_confirm_hypothesis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tools = RepositoryTools(Path(directory))
            result = CommandResult(
                command=["python", "-m", "pytest", "--unknown-option"],
                returncode=4,
                stdout="",
                stderr="pytest: error: unrecognized arguments",
            )
            hypothesis = Hypothesis(
                summary="bug",
                root_file="app/pricing.py",
                confidence=0.8,
                evidence_ids=["evidence-1"],
            )

            supported = _test_failure_supports_hypothesis(result, hypothesis, tools)

        self.assertFalse(supported)

    def test_failure_in_different_file_does_not_confirm_wrong_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tests").mkdir()
            (root / "tests" / "test_calculator.py").write_text(
                "from app.calculator import divide\n", encoding="utf-8"
            )
            result = CommandResult(
                command=["python", "-m", "pytest", "tests/test_calculator.py", "-q"],
                returncode=1,
                stdout="FAILED tests/test_calculator.py::test_divide",
                stderr="1 failed",
            )
            hypothesis = Hypothesis(
                summary="wrong file",
                root_file="app/formatting.py",
                confidence=0.8,
                evidence_ids=["evidence-1"],
            )

            supported = _test_failure_supports_hypothesis(
                result, hypothesis, RepositoryTools(root)
            )

        self.assertFalse(supported)

    def test_failure_supports_transitively_imported_root_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app" / "permissions").mkdir(parents=True)
            (root / "tests").mkdir()
            (root / "app" / "api.py").write_text(
                "from app.permissions.cache import cached_role\n", encoding="utf-8"
            )
            (root / "app" / "permissions" / "cache.py").write_text(
                "def cached_role(): return 'admin'\n", encoding="utf-8"
            )
            (root / "app" / "service.py").write_text(
                "from app.permissions.cache import cached_role\n", encoding="utf-8"
            )
            (root / "app" / "api.py").write_text(
                "from app.service import read_role\n", encoding="utf-8"
            )
            (root / "tests" / "test_access.py").write_text(
                "from app.api import get_dashboard\n", encoding="utf-8"
            )
            result = CommandResult(
                command=["python", "-m", "pytest", "-q"],
                returncode=1,
                stdout="FAILED tests/test_access.py::test_role",
                stderr="1 failed",
            )
            hypothesis = Hypothesis(
                summary="cache key omits tenant", root_file="app/permissions/cache.py",
                confidence=0.9, evidence_ids=["evidence-1"],
            )

            supported = _test_failure_supports_hypothesis(
                result, hypothesis, RepositoryTools(root)
            )

        self.assertTrue(supported)
    @patch("repomedic.dashscope.urllib.request.urlopen")
    def test_client_classifies_api_timeout(self, urlopen: Any) -> None:
        urlopen.side_effect = TimeoutError("timed out")
        client = DashScopeClient(
            "key", "https://example.test", "model", max_retries=0, retry_delay=0
        )

        with self.assertRaises(ModelAPIError) as raised:
            client.complete([{"role": "user", "content": "test"}])

        self.assertEqual(raised.exception.code, "api_timeout")

    @patch("repomedic.dashscope.urllib.request.urlopen")
    def test_client_retries_temporary_connection_error(self, urlopen: Any) -> None:
        response = io.BytesIO(
            json.dumps(
                {
                    "choices": [{"message": {"role": "assistant", "content": "{}"}}],
                    "usage": {"total_tokens": 3},
                }
            ).encode()
        )
        urlopen.side_effect = [urllib.error.URLError("temporary"), response]
        client = DashScopeClient("key", "https://example.test", "model", retry_delay=0)

        message, tokens = client.complete([{"role": "user", "content": "test"}])

        self.assertEqual(message["content"], "{}")
        self.assertEqual(tokens, 3)
        self.assertEqual(urlopen.call_count, 2)

    def test_json_content_accepts_explanation_before_fenced_json(self) -> None:
        content = '分析完成。\n```json\n{"hypotheses": []}\n```'
        self.assertEqual(_json_content({"content": content}), '{"hypotheses": []}')

    def test_token_budget_stops_additional_model_work(self) -> None:
        client = FakeClient(
            [
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "search_terms": ["total_price"],
                            "test_commands": [],
                            "rationale": "test",
                        }
                    ),
                }
            ]
        )
        audit = AuditTrail(PROMPT_VERSION)
        backend = DashScopeBackend(client, audit=audit, max_total_tokens=5)
        with tempfile.TemporaryDirectory() as directory:
            request = DiagnosisInput(repository=Path(directory), issue="prices concatenate")
            with self.assertRaises(AgentRunError) as raised:
                backend.triage(request)

        self.assertEqual(raised.exception.code, "token_limit")
        self.assertEqual(audit.snapshot().token_usage, 10)

    def test_unknown_tool_is_logged_and_model_can_continue(self) -> None:
        client = FakeClient(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "bad-1",
                            "type": "function",
                            "function": {"name": "delete_repository", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": '{"hypotheses": []}',
                },
            ]
        )
        audit = AuditTrail(PROMPT_VERSION)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = DiagnosisInput(repository=root, issue="unknown failure")
            evidence, hypotheses, _ = DashScopeBackend(client, audit=audit).investigate(
                request,
                DiagnosisPlan(search_terms=["failure"], rationale="test"),
                "failure",
                RepositoryTools(root),
            )

        snapshot = audit.snapshot()
        self.assertEqual(evidence, [])
        self.assertEqual(hypotheses, [])
        self.assertEqual(snapshot.attempted_tool_calls, 1)
        self.assertEqual(snapshot.invalid_tool_calls, 1)

    def test_triage_parses_structured_plan(self) -> None:
        client = FakeClient(
            [
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "search_terms": ["total_price"],
                            "test_commands": ["python -m pytest -q"],
                            "rationale": "根据报错函数定位",
                        }
                    ),
                }
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            request = DiagnosisInput(repository=Path(directory), issue="total_price returns text")
            plan, tokens = DashScopeBackend(client).triage(request)
        self.assertEqual(plan.search_terms, ["total_price"])
        self.assertEqual(plan.test_commands, [["python", "-m", "pytest", "-q"]])
        self.assertEqual(tokens, 10)

    def test_triage_repairs_invalid_json_once_without_tools(self) -> None:
        client = FakeClient(
            [
                {"role": "assistant", "content": "not json"},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "search_terms": ["total_price"],
                            "test_commands": [],
                            "rationale": "repaired",
                        }
                    ),
                },
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            request = DiagnosisInput(repository=Path(directory), issue="prices concatenate")
            plan, _ = DashScopeBackend(client).triage(request)

        self.assertEqual(plan.search_terms, ["total_price"])
        self.assertIsNone(client.calls[1]["tools"])

    def test_single_agent_uses_one_tool_loop(self) -> None:
        client = FakeClient(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "search-1",
                            "type": "function",
                            "function": {
                                "name": "search_repository",
                                "arguments": '{"pattern":"total_price"}',
                            },
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "test-1",
                            "type": "function",
                            "function": {
                                "name": "run_command",
                                "arguments": '{"command":["python","-m","pytest","-q"]}',
                            },
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "hypotheses": [
                                {
                                    "summary": "string concatenation",
                                    "root_file": "pricing.py",
                                    "confidence": 0.9,
                                    "suggested_patch": "",
                                }
                            ],
                            "decisions": [
                                {
                                    "hypothesis_index": 0,
                                    "confirmed": True,
                                    "notes": "test reproduced",
                                }
                            ],
                        }
                    ),
                },
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pricing.py").write_text(
                "def total_price(left, right):\n    return left + right\n", encoding="utf-8"
            )
            request = DiagnosisInput(repository=root, issue="prices concatenate")
            evidence, hypotheses, decisions, tokens = DashScopeBackend(
                client
            ).diagnose_single_agent(request, FakeRepositoryTools(root))

        self.assertTrue(evidence)
        self.assertEqual(hypotheses[0].root_file, "pricing.py")
        self.assertTrue(decisions[0].confirmed)
        self.assertEqual(tokens, 30)
        self.assertEqual(
            client.calls[0]["tool_choice"],
            {"type": "function", "function": {"name": "search_repository"}},
        )

    def test_investigation_forces_search_and_binds_real_evidence(self) -> None:
        client = FakeClient(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "search_repository",
                                "arguments": '{"pattern":"total_price","limit":8}',
                            },
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "hypotheses": [
                                {
                                    "summary": "字符串价格被直接拼接",
                                    "root_file": "pricing.py",
                                    "confidence": 0.9,
                                    "suggested_patch": "先转换为数值",
                                }
                            ]
                        }
                    ),
                },
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pricing.py").write_text(
                "def total_price(left, right):\n    return left + right\n", encoding="utf-8"
            )
            request = DiagnosisInput(repository=root, issue="total_price concatenates strings")
            evidence, hypotheses, tokens = DashScopeBackend(client).investigate(
                request,
                DiagnosisPlan(search_terms=["total_price"], rationale="test"),
                "total_price",
                RepositoryTools(root),
            )
        self.assertEqual(evidence[0].path, "pricing.py")
        self.assertEqual(hypotheses[0].root_file, "pricing.py")
        self.assertEqual(tokens, 20)
        self.assertEqual(
            client.calls[0]["tool_choice"],
            {"type": "function", "function": {"name": "search_repository"}},
        )

    def test_verification_requires_an_executed_command(self) -> None:
        client = FakeClient(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-2",
                            "type": "function",
                            "function": {
                                "name": "run_command",
                                "arguments": '{"command":["python","-m","pytest","-q"]}',
                            },
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "decisions": [
                                {
                                    "hypothesis_index": 0,
                                    "confirmed": True,
                                    "notes": "针对性测试复现失败",
                                }
                            ]
                        }
                    ),
                },
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = DiagnosisInput(repository=root, issue="calculation fails")
            evidence = [
                Evidence(id="evidence-1", kind="source", path="pricing.py", excerpt="return")
            ]
            hypotheses = [
                Hypothesis(
                    summary="bad conversion",
                    root_file="pricing.py",
                    confidence=0.8,
                    evidence_ids=["evidence-1"],
                )
            ]
            decisions, tokens = DashScopeBackend(client).verify(
                request, hypotheses, evidence, FakeRepositoryTools(root)
            )
        self.assertTrue(decisions[0].confirmed)
        self.assertEqual(decisions[0].commands[0], ["python", "-m", "pytest", "-q"])
        self.assertIn("实际命令返回码：[1]", decisions[0].notes)
        self.assertEqual(tokens, 20)
        self.assertEqual(
            client.calls[0]["tool_choice"],
            {"type": "function", "function": {"name": "run_command"}},
        )


if __name__ == "__main__":
    unittest.main()
