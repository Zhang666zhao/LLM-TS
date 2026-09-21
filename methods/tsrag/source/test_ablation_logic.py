#!/usr/bin/env python3
"""Fast deterministic checks for the ablation routing equations."""

import torch
import torch.nn.functional as F

from inference_reproduce import intervention_batches


def main() -> None:
    raw = torch.tensor([[[0.0], [1.0], [-1.0]]])
    official = F.softmax(torch.sigmoid(raw), dim=1)
    raw_softmax = F.softmax(raw, dim=1)
    uniform = torch.full_like(raw, 1.0 / 3.0)
    assert torch.allclose(official.sum(dim=1), torch.ones(1, 1))
    assert torch.allclose(raw_softmax.sum(dim=1), torch.ones(1, 1))
    assert torch.allclose(uniform.sum(dim=1), torch.ones(1, 1))
    assert raw_softmax.max() > official.max(), (raw_softmax, official)

    distances = torch.tensor([[1.0, 2.0, 3.0]])
    normalized = (distances - distances.mean(dim=1, keepdim=True)) / distances.std(
        dim=1, keepdim=True, unbiased=False
    )
    beta_raw = torch.tensor(0.541324854612918, requires_grad=True)
    corrected = -F.softplus(beta_raw) * normalized
    assert corrected[0, 0] > corrected[0, 1] > corrected[0, 2]
    corrected.sum().backward()
    assert beta_raw.grad is not None and torch.isfinite(beta_raw.grad)

    batches = []
    for start in (0, 2, 4):
        retrieved = torch.arange(start, start + 2).view(2, 1, 1).float()
        batches.append([torch.zeros(2, 1), torch.zeros(2, 1), None, None, retrieved, torch.zeros(2, 1)])
    shifted = list(intervention_batches(batches, "shuffle_future"))
    original_ids = torch.cat([batch[4][:, 0, 0] for batch in batches])
    shifted_ids = torch.cat([batch[4][:, 0, 0] for batch in shifted])
    assert torch.equal(shifted_ids, torch.roll(original_ids, shifts=-1))
    assert not torch.any(shifted_ids == original_ids)
    print("ablation logic checks passed")


if __name__ == "__main__":
    main()
