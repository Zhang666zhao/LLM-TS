# Method contract

Each method owns one directory containing:

- `method.json`: stable method ID, display name, implementation entry point,
  and provenance;
- `adapter.py`: a CLI accepting the common dataset/path/run arguments;
- source code and method-specific dependencies;
- a README describing installation and model assets.

The common launcher calls adapters as subprocesses. Adapters may use different
environments internally, but every completed run must emit a standard
`metrics.json` plus optional `variable_metrics.csv`. New methods can start from
`template/` without changing the evaluation package.
