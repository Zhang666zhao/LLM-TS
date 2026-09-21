#!/usr/bin/env python3
"""Replace this placeholder when adding a new method."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-manifest-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.parse_args()
    raise SystemExit("Template adapter is not implemented; copy and replace it")


if __name__ == "__main__":
    main()
