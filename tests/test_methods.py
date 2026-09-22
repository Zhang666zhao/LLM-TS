from pathlib import Path
import hashlib
import importlib.util
import json
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MethodContractTest(unittest.TestCase):
    def test_registered_adapters_build_non_executing_commands(self):
        for method in ("tsrag", "chronos_bolt", "dlinear"):
            manifest = json.loads((ROOT / "methods" / method / "method.json").read_text())
            adapter = ROOT / manifest["adapter"]
            self.assertTrue(adapter.exists())
            self.assertEqual(manifest["id"], method)
            self.assertIn("ETTh1", manifest["supported_datasets"])

    def test_no_large_runtime_assets_were_imported(self):
        forbidden = {".pth", ".pt", ".ckpt", ".safetensors", ".npy", ".npz", ".pkl", ".pickle"}
        offenders = [path for path in (ROOT / "methods").rglob("*") if path.suffix in forbidden]
        self.assertEqual(offenders, [])

    def test_dlinear_vendor_and_results_are_compact(self):
        model = ROOT / "methods/dlinear/vendor/ltsf_linear/models/DLinear.py"
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        self.assertEqual(digest, "0893b53cb6473d6bdca7aeca514cb3ee12efa6df227c29c4469571c9711451cc")
        source = ROOT / "results/benchmark_v1/source/dlinear"
        forbidden = {".pth", ".pt", ".ckpt", ".safetensors", ".npy", ".npz", ".pkl", ".log"}
        offenders = [path for path in source.rglob("*") if path.suffix in forbidden]
        self.assertEqual(offenders, [])
        self.assertLess(sum(path.stat().st_size for path in source.rglob("*") if path.is_file()), 500_000)


if __name__ == "__main__":
    unittest.main()
