from __future__ import annotations

import unittest

from redisdoctor.cli import build_parser as build_redisdoctor_parser
from repomedic.cli import build_parser as build_repomedic_parser
from repomedic.security import CommandPolicy, UnsafeOperation


class ChineseUiTest(unittest.TestCase):
    def test_repomedic_help_is_chinese(self) -> None:
        help_text = build_repomedic_parser().format_help()
        self.assertIn("证据优先", help_text)
        self.assertIn("待诊断的Python仓库路径", help_text)

    def test_redisdoctor_help_is_chinese(self) -> None:
        help_text = build_redisdoctor_parser().format_help()
        self.assertIn("版本感知", help_text)
        self.assertIn("Redis Markdown文档", help_text)

    def test_security_error_is_chinese(self) -> None:
        with self.assertRaisesRegex(UnsafeOperation, "白名单"):
            CommandPolicy().validate(["powershell", "-Command", "Get-ChildItem"])


if __name__ == "__main__":
    unittest.main()
