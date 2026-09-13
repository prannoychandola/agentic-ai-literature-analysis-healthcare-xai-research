"""Verify repository-relative paths do not depend on the terminal directory."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_xai.config import PROJECT_ROOT


class PathTests(unittest.TestCase):
    def test_project_root_contains_expected_files(self) -> None:
        self.assertEqual(PROJECT_ROOT.name, "agentic-ai-xai-research")
        self.assertTrue((PROJECT_ROOT / "README.md").is_file())
        self.assertTrue((PROJECT_ROOT / "data" / "corpus.csv").is_file())
