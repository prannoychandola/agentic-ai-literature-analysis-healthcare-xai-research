"""Ensure all package modules import without network or model execution."""

import importlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class PackageImportTests(unittest.TestCase):
    def test_package_modules_import(self) -> None:
        for name in (
            "agentic_xai",
            "agentic_xai.config",
            "agentic_xai.utils",
            "agentic_xai.embeddings",
            "agentic_xai.retrieval",
            "agentic_xai.clustering",
            "agentic_xai.evaluation",
            "agentic_xai.ablation",
            "agentic_xai.stability",
            "agentic_xai.gap_analysis",
        ):
            with self.subTest(module=name):
                importlib.import_module(name)
