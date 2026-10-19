# Direction-conditioned forecasting research

## Scope (2026-10-19)
- Dedicated branch: research/direction-conditioned-forecasting; separate worktree.
- Current authorization: repository setup, historical documentation and simple plan only.
- Do not launch training, paid API calls, upload time-series data, or change environments without a separately specified scope and budget.
- Fixed interface: observed multivariate history -> per-variable labels in {-1,0,+1}.
- New label definitions and thresholds are configurable research choices. Historical oracle_v1 used tau=0.5, epsilon=0.001, ddof=0; NEVER rewrite those results as another threshold.
- No future values in inference inputs, feature construction, retrieval selection, or prompt tuning. Training labels may use their own future targets.
- Select prompt/threshold on train/validation only; preserve chronological boundaries and evaluate frozen choices on test.
- Preserve A-L1, A-L2, B-L1, B-L2 and respective matched-zero controls separately.
- Historical CPU/backbone seed2021, actual CUDA2022; any correction is a NEW experiment version.
- Local historical source copies are not portable/reverified reproductions. Remote SSH authentication blocked fresh audit on2026-10-19; see provenance.
- No raw data, checkpoints, label caches, prediction arrays, credentials or environment folders in Git.
- Preserve existing benchmark code and unrelated edits; do not refactor frozen copies.
