#!/usr/bin/env python3
"""Preflight checks for Training AutoResearch runs."""

import argparse
import fnmatch
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union

RUN_RE = re.compile(r"^[0-9]{8}_[A-Za-z0-9][A-Za-z0-9._-]*_[A-Za-z0-9._-]+_[A-Za-z0-9._-]+$")


def git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )


def parse_simple_yaml_lists(path: Path) -> Dict[str, Union[List[str], str]]:
    """Tiny parser for the scaffolded boundary YAML; avoids external deps."""
    data = {}  # type: Dict[str, Union[List[str], str]]
    current = None  # type: Optional[str]
    parent = None  # type: Optional[str]
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if not raw.startswith(" ") and ":" in line:
            key, value = line.split(":", 1)
            current = key.strip()
            parent = current
            value = value.strip()
            if value:
                data[current] = value
            else:
                data[current] = []
        elif raw.startswith("  ") and not raw.startswith("    ") and ":" in line and parent:
            key, value = line.strip().split(":", 1)
            data[f"{parent}.{key.strip()}"] = value.strip()
            current = f"{parent}.{key.strip()}"
        elif current and line.strip().startswith("- "):
            data.setdefault(current, [])
            value = line.strip()[2:].strip()
            if isinstance(data[current], list):
                data[current].append(value)
    return data


def changed_files(root: Path) -> List[str]:
    result = git(root, "status", "--short")
    if result.returncode != 0:
        raise SystemExit("Not a git repository. Initialize git and commit baseline before training.")
    files = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        files.append(line[3:].strip())
    return files


def committed_changes(root: Path, baseline: str) -> List[str]:
    result = git(root, "diff", "--name-only", f"{baseline}..HEAD")
    if result.returncode != 0:
        raise SystemExit(f"Cannot diff benchmark_baseline_commit {baseline}..HEAD.")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def matches_any(path: str, patterns: List[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def check_run_name(run_name: str) -> None:
    if not RUN_RE.match(run_name):
        raise SystemExit("Invalid run name. Use YYYYMMDD_model_keychange_metric.")
    if "/" in run_name or "\\" in run_name or ".." in run_name or Path(run_name).is_absolute():
        raise SystemExit("Invalid run name: path separators, '..', and absolute paths are forbidden.")


def required_field(config: Dict[str, Union[List[str], str]], key: str) -> str:
    value = config.get(key, "")
    if isinstance(value, list):
        raise SystemExit(f"Boundary field {key} must be a scalar.")
    if not value or value.strip().upper() == "TODO":
        raise SystemExit(f"Boundary field {key} is required and cannot be TODO.")
    return value.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--data-version", default="")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-protected-changes", action="store_true")
    args = parser.parse_args()

    check_run_name(args.run_name)
    root = Path(args.root).resolve()
    boundary = root / "agent_boundary.yaml"
    if not boundary.exists():
        raise SystemExit("Missing agent_boundary.yaml. Create boundaries before AutoResearch.")
    if not args.data_version.strip() and not (root / "data_version.txt").exists():
        raise SystemExit("Missing data version. Pass --data-version or create data_version.txt.")

    config = parse_simple_yaml_lists(boundary)
    required_field(config, "research_goal")
    required_field(config, "official_benchmark.data")
    required_field(config, "official_benchmark.split")
    required_field(config, "official_benchmark.label")
    required_field(config, "official_benchmark.primary_metric")
    direction = required_field(config, "official_benchmark.metric_direction")
    if direction not in ("max", "min"):
        raise SystemExit("official_benchmark.metric_direction must be max or min.")
    required_field(config, "approval_owner")
    baseline_commit = required_field(config, "benchmark_baseline_commit")

    head = git(root, "rev-parse", "HEAD")
    if head.returncode != 0:
        raise SystemExit("Git HEAD unavailable. Commit project state before training.")

    files = changed_files(root)
    if files and not args.allow_dirty:
        raise SystemExit("Dirty worktree. Checkpoint or commit before run:\n" + "\n".join(files))

    immutable = config.get("immutable", [])
    review_required = config.get("review_required", [])
    benchmark_versioned = config.get("benchmark_versioned", [])
    if not isinstance(immutable, list):
        immutable = []
    if not isinstance(review_required, list):
        review_required = []
    if not isinstance(benchmark_versioned, list):
        benchmark_versioned = []

    changed_since_baseline = sorted(set(files + committed_changes(root, baseline_commit)))
    protected = [
        path for path in files
        if matches_any(path, immutable) and not matches_any(path, benchmark_versioned)
    ]
    protected.extend(
        path for path in changed_since_baseline
        if matches_any(path, immutable) and not matches_any(path, benchmark_versioned)
    )
    review_changes = [
        path for path in changed_since_baseline
        if matches_any(path, review_required) and not matches_any(path, benchmark_versioned)
    ]
    if protected and not args.allow_protected_changes:
        raise SystemExit(
            "Protected benchmark files are dirty. Ask for benchmark-version approval first:\n"
            + "\n".join(sorted(set(protected)))
        )
    if review_changes and not args.allow_protected_changes:
        raise SystemExit(
            "Review-required files changed since benchmark baseline. Confirm no benchmark drift:\n"
            + "\n".join(sorted(set(review_changes)))
        )

    print("Preflight passed")


if __name__ == "__main__":
    main()
