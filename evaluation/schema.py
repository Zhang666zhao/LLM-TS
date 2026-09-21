"""Schema and validation for comparable variable-level metrics."""

from __future__ import annotations

from dataclasses import dataclass
import math


REQUIRED_COLUMNS = (
    "benchmark_id", "method", "dataset", "frequency", "feature_id",
    "variable", "seed", "context_length", "prediction_length",
    "test_windows", "mse", "mae", "source",
)


@dataclass(frozen=True)
class VariableMetric:
    benchmark_id: str
    method: str
    dataset: str
    frequency: str
    feature_id: int
    variable: str
    seed: int
    context_length: int
    prediction_length: int
    test_windows: int
    mse: float
    mae: float
    source: str

    def validate(self) -> None:
        for name in ("benchmark_id", "method", "dataset", "variable", "source"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        for name in ("feature_id", "seed", "context_length", "prediction_length", "test_windows"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.context_length == 0 or self.prediction_length == 0 or self.test_windows == 0:
            raise ValueError("context_length, prediction_length, and test_windows must be positive")
        for name in ("mse", "mae"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {name: getattr(self, name) for name in REQUIRED_COLUMNS}
