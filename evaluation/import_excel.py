"""Import the completed TS-RAG/Chronos-Bolt workbook without inference."""

from __future__ import annotations

import hashlib
from pathlib import Path

from evaluation.io import write_json, write_variable_metrics
from evaluation.metrics import benchmark_summaries, dataset_summaries, pairwise_comparison
from evaluation.schema import VariableMetric


DATASET_NAMES = {
    "ETTh1": "ETTh1", "ETTh2": "ETTh2", "ETTm1": "ETTm1", "ETTm2": "ETTm2",
    "Weather": "weather", "Electricity": "electricity", "Exchange": "exchange_rate",
}
METHOD_NAMES = {"TS-RAG": "tsrag", "Chronos-Bolt": "chronos_bolt"}
SUMMARY_DATASET_NAMES = {
    "ETTh1": "ETTh1", "ETTh2": "ETTh2", "ETTm1": "ETTm1", "ETTm2": "ETTm2",
    "Weather": "weather", "Electricity": "electricity", "Exchange": "exchange_rate",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_workbook(source: Path, output_dir: Path) -> dict[str, object]:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError("Install openpyxl to import the source workbook") from exc

    workbook = load_workbook(source, data_only=True, read_only=False)
    if "Long data" not in workbook.sheetnames:
        raise ValueError("Workbook is missing the 'Long data' sheet")
    sheet = workbook["Long data"]
    headers = [cell.value for cell in sheet[1]]
    expected = ["Dataset", "Frequency", "Feature ID", "Variable", "Model", "MSE", "MAE", "Test windows", "Horizon"]
    if headers != expected:
        raise ValueError(f"Unexpected Long data columns: {headers}")

    rows: list[VariableMetric] = []
    for values in sheet.iter_rows(min_row=2, values_only=True):
        if all(value is None for value in values):
            continue
        dataset, frequency, feature_id, variable, model, mse, mae, windows, horizon = values
        try:
            canonical_dataset = DATASET_NAMES[str(dataset)]
            canonical_method = METHOD_NAMES[str(model)]
        except KeyError as exc:
            raise ValueError(f"Unknown workbook label: {exc.args[0]}") from exc
        rows.append(VariableMetric(
            benchmark_id="benchmark_v1", method=canonical_method,
            dataset=canonical_dataset, frequency=str(frequency), feature_id=int(feature_id),
            variable=str(variable), seed=2021, context_length=512,
            prediction_length=int(horizon), test_windows=int(windows),
            mse=float(mse), mae=float(mae), source=source.name,
        ))

    if len(rows) != 756:
        raise ValueError(f"Expected 756 variable-method rows, found {len(rows)}")
    keys = {(row.method, row.dataset, row.feature_id) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("Duplicate method/dataset/feature rows in workbook")

    dataset_rows = dataset_summaries(rows)
    by_dataset_method = {
        (str(row["dataset"]), str(row["method"])): row for row in dataset_rows
    }
    summary_sheet = workbook["Summary"]
    for values in summary_sheet.iter_rows(min_row=7, max_row=13, values_only=True):
        dataset_label, variable_count, total_windows = values[:3]
        dataset = SUMMARY_DATASET_NAMES[str(dataset_label)]
        expected = {
            "tsrag": (float(values[3]), float(values[7])),
            "chronos_bolt": (float(values[4]), float(values[8])),
        }
        for method, (mse, mae) in expected.items():
            actual = by_dataset_method[(dataset, method)]
            if int(actual["variables"]) != int(variable_count):
                raise ValueError(f"Variable count mismatch for {method}/{dataset}")
            if int(actual["test_windows"]) != int(total_windows):
                raise ValueError(f"Test-window count mismatch for {method}/{dataset}")
            if abs(float(actual["mse"]) - mse) > 1e-12 or abs(float(actual["mae"]) - mae) > 1e-12:
                raise ValueError(f"Summary metric mismatch for {method}/{dataset}")
    benchmark_rows = benchmark_summaries(dataset_rows)
    comparisons = pairwise_comparison(dataset_rows, baseline="chronos_bolt")
    write_variable_metrics(output_dir / "variable_metrics.csv", rows)
    write_json(output_dir / "dataset_metrics.json", dataset_rows)
    write_json(output_dir / "summary.json", {
        "benchmark_id": "benchmark_v1",
        "schema_version": "1.0",
        "source": source.name,
        "source_sha256": sha256(source),
        "row_count": len(rows),
        "benchmark": benchmark_rows,
        "comparisons": comparisons,
    })
    return {"rows": rows, "datasets": dataset_rows, "benchmark": benchmark_rows}
