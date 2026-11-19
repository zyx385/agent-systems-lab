from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from repomedic.security import CommandPolicy, UnsafeOperation, safe_repository_path
from repomedic.tools import RepositoryTools


class RepositoryToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "sample.py").write_text(
            "def calculate_total(items):\n    return sum(items)\n", encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_safe_path_rejects_traversal(self) -> None:
        with self.assertRaises(UnsafeOperation):
            safe_repository_path(self.root, "../secret.txt")

    def test_command_policy_rejects_shell_metacharacters(self) -> None:
        with self.assertRaises(UnsafeOperation):
            CommandPolicy().validate(["python", "-m", "pytest", ";", "whoami"])

    def test_command_policy_rejects_unknown_command(self) -> None:
        with self.assertRaises(UnsafeOperation):
            CommandPolicy().validate(["powershell", "-Command", "Get-ChildItem"])

    def test_search_and_call_graph(self) -> None:
        tools = RepositoryTools(self.root)
        evidence = tools.search("calculate_total")
        graph = tools.call_graph("sample.py")
        self.assertEqual(evidence[0].path, "sample.py")
        self.assertEqual(graph.definitions["calculate_total"], ["sum"])

    def test_write_requires_approval(self) -> None:
        tools = RepositoryTools(self.root)
        with self.assertRaises(UnsafeOperation):
            tools.apply_text("new.py", "pass\n", approved=False)
        self.assertFalse((self.root / "new.py").exists())


if __name__ == "__main__":
    unittest.main()
