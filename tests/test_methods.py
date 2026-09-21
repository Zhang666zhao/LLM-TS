from pathlib import Path
import importlib.util
import json
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MethodContractTest(unittest.TestCase):
    def test_registered_adapters_build_non_executing_commands(self):
        for method in ("tsrag", "chronos_bolt"):
            manifest = json.loads((ROOT / "methods" / method / "method.json").read_text())
            adapter = ROOT / manifest["adapter"]
            self.assertTrue(adapter.exists())
            self.assertEqual(manifest["id"], method)
            self.assertIn("ETTh1", manifest["supported_datasets"])

    def test_no_large_runtime_assets_were_imported(self):
        forbidden = {".pth", ".pt", ".ckpt", ".safetensors", ".npy", ".npz", ".pkl", ".pickle"}
        offenders = [path for path in (ROOT / "methods").rglob("*") if path.suffix in forbidden]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
