# TS-RAG method snapshot

`source/` is a code-only snapshot of the work on commit `a4bacfa` from branch
`research/tsrag-retrieval-offline` on GPU_Foundation. Data, model weights,
checkpoints, artifacts, runs, Python caches, and the nested Git directory were
excluded.

The upstream project is <https://github.com/UConn-DSIS/TS-RAG>. Its license is
preserved in this directory. `pyproject.server.toml` and
`requirements.server.txt` record the server environment requirements at import
time. The snapshot itself is kept intact; portable paths are supplied through
the adapter rather than written into upstream source files.

Use the shared launcher from the repository root:

```bash
python scripts/run_method.py --method tsrag --dataset ETTh1 --dry-run
```
