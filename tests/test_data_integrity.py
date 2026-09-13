"""Read committed artifacts without rerunning or modifying any experiment."""

import csv
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_xai.config import DATA_DIR, OUTPUTS_DIR


class DataIntegrityTests(unittest.TestCase):
    def test_corpus_and_embedding_metadata(self) -> None:
        with (DATA_DIR / "corpus.csv").open(newline="", encoding="utf-8") as handle:
            corpus = list(csv.DictReader(handle))
        self.assertEqual(len(corpus), 187)
        self.assertEqual(sum(bool(row["abstract"].strip()) for row in corpus), 175)
        matrix = np.load(OUTPUTS_DIR / "embeddings.npy", allow_pickle=False)
        self.assertEqual(matrix.shape, (175, 384))

    def test_required_json_outputs_are_readable(self) -> None:
        for filename in (
            "corpus_stats.json",
            "clustering_results.json",
            "baseline_results.json",
            "ablation_results.json",
            "stability_results.json",
            "gap_analysis.json",
        ):
            with self.subTest(filename=filename):
                with (OUTPUTS_DIR / filename).open(encoding="utf-8") as handle:
                    self.assertIsInstance(json.load(handle), dict)
