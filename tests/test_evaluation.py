from pathlib import Path
import json
import tempfile
import unittest

from evaluation.import_excel import import_workbook
from evaluation.io import read_variable_metrics
from evaluation.metrics import benchmark_summaries, dataset_summaries
from evaluation.schema import VariableMetric


ROOT = Path(__file__).resolve().parents[1]


class ExistingWorkbookTest(unittest.TestCase):
    def test_import_and_reconcile(self):
        source = ROOT / "results/benchmark_v1/source/TSRAG_ChronosBolt_variable_metrics.xlsx"
        dlinear = ROOT / "results/benchmark_v1/source/dlinear/variable_metrics.csv"
        with tempfile.TemporaryDirectory() as directory:
            result = import_workbook(source, Path(directory), dlinear)
            self.assertEqual(len(result["rows"]), 1134)
            self.assertEqual(len(result["datasets"]), 21)
            by_method = {row["method"]: row for row in result["benchmark"]}
            self.assertAlmostEqual(by_method["tsrag"]["mse"], 0.19389417357968552, places=12)
            self.assertAlmostEqual(by_method["chronos_bolt"]["mse"], 0.2007255045682274, places=12)
            self.assertAlmostEqual(by_method["dlinear"]["mse"], 0.19850722644067428, places=12)
            self.assertAlmostEqual(by_method["dlinear"]["mae"], 0.2748751095859167, places=12)
            etth1 = next(
                row for row in result["datasets"]
                if row["method"] == "tsrag" and row["dataset"] == "ETTh1"
            )
            self.assertEqual(etth1["variables"], 7)
            self.assertEqual(etth1["test_windows"], 19719)
            self.assertAlmostEqual(etth1["mse"], 0.3555922504835311, places=12)
            summary = json.loads((Path(directory) / "summary.json").read_text())
            self.assertEqual(summary["row_count"], 1134)

    def test_committed_csv_contract(self):
        path = ROOT / "results/benchmark_v1/variable_metrics.csv"
        if not path.exists():
            self.skipTest("generated result not created yet")
        rows = read_variable_metrics(path)
        self.assertEqual(len(rows), 1134)
        datasets = dataset_summaries(rows)
        self.assertEqual(len(datasets), 21)
        self.assertEqual(len(benchmark_summaries(datasets)), 3)

    def test_distinct_seeds_are_not_mixed(self):
        base = dict(
            benchmark_id="benchmark_v1", method="tsrag", dataset="ETTh1",
            frequency="1 hour", feature_id=0, variable="HUFL",
            context_length=512, prediction_length=64, test_windows=10,
            mse=0.2, mae=0.3, source="test",
        )
        summaries = dataset_summaries([
            VariableMetric(seed=2021, **base),
            VariableMetric(seed=2022, **base),
        ])
        self.assertEqual(len(summaries), 2)
        self.assertEqual({row["seed"] for row in summaries}, {2021, 2022})
        self.assertTrue(all(row["variables"] == 1 for row in summaries))


if __name__ == "__main__":
    unittest.main()
