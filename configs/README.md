# Configuration

- `benchmark.yaml` defines the comparison contract shared by every method.
- `datasets.yaml` records canonical dataset names and frequencies.
- `paths.example.yaml` is a portable template. Copy it to
  `paths.local.yaml`; the local file is ignored by Git.
- `experiments/` contains reproducible experiment definitions.

Method-specific research configurations remain inside each method directory.
