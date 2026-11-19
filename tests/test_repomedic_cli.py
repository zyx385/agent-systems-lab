from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from repomedic.cli import main
from repomedic.dashscope import AgentRunError


class RepoMedicCliTest(unittest.TestCase):
    @patch("repomedic.cli.DashScopeBackend.from_env")
    def test_model_error_returns_structured_log_without_crashing(self, from_env: MagicMock) -> None:
        backend = MagicMock()
        backend.diagnose_single_agent.side_effect = AgentRunError(
            "invalid_json", "模型回复中没有JSON"
        )
        from_env.return_value = backend
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit_path = root / "audit.json"
            exit_code = main(
                [
                    str(root),
                    "--issue",
                    "model returns invalid JSON",
                    "--backend",
                    "dashscope",
                    "--baseline",
                    "--audit-log",
                    str(audit_path),
                    "--json",
                ]
            )
            payload = json.loads(audit_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["error"]["failure_reason"], "invalid_json")
        self.assertIn("audit", payload)


if __name__ == "__main__":
    unittest.main()
