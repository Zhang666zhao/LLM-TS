#!/usr/bin/env python3
"""Fast deterministic tests for retrieval-ablation artifact logic."""

import numpy as np

from retrieval_ablation_artifacts import adaptive_filtered_search, qwen_text, splitmix64, standardize_rows
from summarize_retrieval_ablation import paired_block_ci


def main() -> None:
    values = np.vstack([np.arange(512, dtype=np.float32), -np.arange(512, dtype=np.float32)])
    normalized = standardize_rows(values)
    assert normalized.shape == (2, 512)
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0, atol=1e-5)
    correlation = normalized @ normalized.T
    assert correlation[0, 1] < -0.999
    signs = np.sign(correlation[0])
    assert signs.tolist() == [1.0, -1.0]
    texts = qwen_text(values)
    for text in texts:
        payload = text.split("Values: ", 1)[1].split()
        assert len(payload) == 512
        assert all(0 <= int(value) <= 255 for value in payload)
    ids = np.arange(1000, dtype=np.uint64)
    first = splitmix64(ids.copy())
    second = splitmix64(ids.copy())
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == len(first)
    retrieved = np.ones((2, 10, 576), dtype=np.float32)
    retrieval_signs = np.ones((2, 10), dtype=np.int8)
    retrieval_signs[:, 1::2] = -1
    signed = retrieved * retrieval_signs[..., None]
    assert np.all(signed[:, 0] == 1) and np.all(signed[:, 1] == -1)
    sample = {
        "sample_id": np.arange(256),
        "feature_id": np.repeat(np.arange(2), 128),
        "mse": np.linspace(0.1, 0.2, 256),
    }
    candidate = {**sample, "mse": sample["mse"] + 0.01}
    first_ci = paired_block_ci(sample, candidate)
    second_ci = paired_block_ci(sample, candidate)
    assert first_ci == second_ci and np.allclose(first_ci[:3], 0.01)

    # The first 80 candidates are exact-context duplicates. Top-64 therefore
    # becomes empty after filtering and must expand to return ten valid rows.
    class FakeIndex:
        ntotal = 128

        def search(self, queries, search_k):
            rows = len(queries)
            ids = np.tile(np.arange(search_k, dtype=np.int64), (rows, 1))
            distances = np.tile(np.arange(search_k, dtype=np.float32), (rows, 1))
            return distances, ids

    database_hashes = np.arange(128, dtype=np.uint64) + 1000
    database_hashes[:80] = 7
    selected, diagnostics = adaptive_filtered_search(
        FakeIndex(),
        np.zeros((1, 4), dtype=np.float32),
        np.asarray([7], dtype=np.uint64),
        database_hashes,
        "bolt",
        initial_search_k=64,
        max_search_k=128,
    )
    selected_ids, selected_distances, selected_signs = selected
    assert selected_ids.tolist() == [list(range(80, 90))]
    assert np.allclose(selected_distances, np.arange(80, 90, dtype=np.float32)[None])
    assert np.all(selected_signs == 1)
    assert diagnostics["final_search_k"] == 128
    assert diagnostics["attempts"][0]["insufficient"] == 1
    print("retrieval ablation logic tests passed")


if __name__ == "__main__":
    main()
