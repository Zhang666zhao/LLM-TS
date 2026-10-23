import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("predict", ROOT / "predict.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class Contracts(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/dao.json").read_text())
        self.config["label"]["recent"] = 2
        self.window = {"window_id": "synthetic", "cutoff": "2020-01-01T02:00:00Z",
                       "timestamps": ["2020-01-01T00:00:00Z", "2020-01-01T01:00:00Z", "2020-01-01T02:00:00Z"],
                       "variables": ["x", "y"], "history": [[1, 2], [2, 2], [3, 2]]}

    def test_history_mandatory_stats_optional(self):
        for enabled in (True, False):
            self.config["include_stats"] = enabled
            result = p.evidence(self.window, self.config)
            self.assertEqual(result["history"], self.window["history"])
            self.assertEqual("historical_statistics" in result, enabled)

    def test_future_fields(self):
        self.window["future"] = [9]
        with self.assertRaises(ValueError):
            p.evidence(self.window, self.config)

    def test_future_timestamp(self):
        self.window["timestamps"][-1] = "2020-01-01T03:00:00Z"
        with self.assertRaises(ValueError):
            p.evidence(self.window, self.config)

    def test_nonfinite(self):
        self.window["history"][0][0] = float("nan")
        with self.assertRaises(ValueError):
            p.evidence(self.window, self.config)

    def test_configurable_thresholds(self):
        self.config["label"].update(lower=-0.2, upper=0.7, version="new")
        self.assertEqual(p.evidence(self.window, self.config)["label_definition"]["upper"], 0.7)

    def test_semantics(self):
        self.config["include_semantics"] = True
        self.window["semantics"] = {"x": {"description": "unknown", "source": "", "verified": False}}
        with self.assertRaises(ValueError):
            p.evidence(self.window, self.config)

    def test_invalid_labels(self):
        for rows in ([], [{"variable": "x", "L1": True, "L2": 0}],
                     [{"variable": "x", "L1": 1, "L2": 0}] * 2):
            with self.assertRaises(ValueError):
                p.validate_response(json.dumps({"labels": rows}), ["x", "y"])

    def test_dry_no_network(self):
        with patch.object(p.ChatCompletions, "complete", side_effect=AssertionError("network")):
            self.assertEqual(p.run(self.window, self.config)["status"], "dry_run")

    def test_mock_send(self):
        raw = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"labels": [
            {"variable": n, "L1": 1, "L2": 0} for n in ("x", "y")]})}}]}
        with patch.object(p.ChatCompletions, "complete", return_value=raw) as call:
            self.assertEqual(p.run(self.window, self.config, send=True)["status"], "valid")
            call.assert_called_once()
        raw["choices"][0]["finish_reason"] = "length"
        with patch.object(p.ChatCompletions, "complete", return_value=raw):
            result = p.run(self.window, self.config, send=True)
        self.assertEqual(result["status"], "invalid_response")
        self.assertNotIn("prediction", result)

    def test_hash_model(self):
        first = p.run(self.window, self.config)["request_hash"]
        self.config["model"] = "aliyun/DeepSeek-v4-pro"
        self.assertNotEqual(first, p.run(self.window, self.config)["request_hash"])


if __name__ == "__main__":
    unittest.main()
