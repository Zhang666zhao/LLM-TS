from pathlib import Path
import json
import tempfile
import unittest

from evaluation.evaluate import validate
from methods.common import normalize_dlinear_metrics, normalize_legacy_metrics


class AdapterSchemaTest(unittest.TestCase):
    def test_legacy_result_is_normalized_and_preserved(self):
        native = {
            "status": "complete", "mse": 0.2, "mae": 0.3,
            "sample_count": 10, "value_count": 640,
            "git_commit": "abc123", "manifest_sha256": "legacy-hash",
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "metrics.json").write_text(json.dumps(native))
            normalize_legacy_metrics(
                output, benchmark_id="benchmark_v1", run_id="run-1",
                method="tsrag", dataset="ETTh1", seed=2021,
                context_length=512, prediction_length=64,
                data_manifest_sha256="contract-hash",
            )
            normalized = json.loads((output / "metrics.json").read_text())
            validate(normalized)
            self.assertEqual(normalized["metrics"], {"mse": 0.2, "mae": 0.3})
            self.assertEqual(normalized["data_manifest_sha256"], "contract-hash")
            self.assertEqual(json.loads((output / "native_metrics.json").read_text()), native)

    def test_dlinear_result_is_normalized_and_preserved(self):
        native = {
            "status": "complete", "mse": 0.2, "mae": 0.3,
            "sample_count_tsrag_semantics": 70, "value_count": 4480,
            "context_length": 512, "prediction_length": 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "metrics.json").write_text(json.dumps(native))
            normalize_dlinear_metrics(
                output, benchmark_id="benchmark_v1", run_id="run-dlinear",
                dataset="ETTh1", seed=2021, data_manifest_sha256="contract-hash",
            )
            normalized = json.loads((output / "metrics.json").read_text())
            validate(normalized)
            self.assertEqual(normalized["method"], "dlinear")
            self.assertEqual(normalized["training_regime"], "supervised_per_dataset")
            self.assertEqual(normalized["sample_count"], 70)


if __name__ == "__main__":
    unittest.main()
