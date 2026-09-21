# New method template

Copy this directory to `methods/<method_id>`, update `method.json`, and
implement `adapter.py`. Do not add dataset copies, checkpoints, or full run
outputs. The adapter must accept the common launcher arguments and write the
standard result schema described in `evaluation/README.md`.
