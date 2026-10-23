# Research failure log

## 2026-10-19 — Oracle-noise preparation import error (zero evaluations)

Resolved after explicit user continuation: fixed v2 code5fcbe20 completed all80
evaluations at23:12:16 CST, original deadline preserved. Independent100-result
audit passed; report reports/etth1_oracle_noise_20261019.md. No failed GPU job
was rerun (v1 failed before dispatch). Keep this failure record.

v1 called train.labels, but the pinned label function is dataset.labels.
Failed during preparation before any GPU inference;0/80 evaluations ran.
Originalcode1af0d71 and failed run/budget retained. Fix737ddfb imports from
dataset and adds read-only check_source.py against all20 old oracle caches.
No automatic queue restart or budget reset; monitor paused pending direction.


## 2026-10-19 — ETTh1 predicted-label downstream gains not established

Fixed40-job suite completed and independently verified: no execution failures.
G1 direct oracle-to-LLM condition substitution worsens MSE versus baseline in
20/20 schemes. G2 predicted-condition retraining improves18/20 versus G1 but
only6/20 versus baseline and7/20 versus matched-zero, generally tiny gains.
DLinear-A large baseline gap also exists with zero conditions, so it is not
evidence that LLM errors alone caused the gap. Distribution mismatch is a
plausible interpretation, not a proven mechanism. Preserve negative evidence;
do not automatically tune prompts/thresholds or rerun based on test results.
See reports/etth1_llm_downstream_20261019.md. Monitoring paused, no new training.

## 2026-10-19 — stride32 completed after recorded rate-limit recovery

Frozen Pro/prompt:89 anchors cover2817 ETTh1 validation windows. New89requests=85 unique+4 repairs, plus4 verified historical caches. Two429 limits required admission pacing3.5s then10s; two joint-label conflicts were retained and repaired with the same prompt. All89 anchors valid in finalstride32_v5;19719 rows saved. Full F1 L1 .4035,L2 .6049; accuracy63.26/63.50%. No successful request repeated. Originaldeadline preserved. See reports/direction_labels_stride32_20261019.md.

## 2026-10-19 — Complete Pro labels, F1 goal not met

Within12-request/30min exploration, compact Flash still truncated; grouped Flash only partially valid. Pro+concise reasoning prompt produced complete labels on2 windows: L1 macroF1 .2424 (all0), L2 .3000; below .5.8 attempts/971s total, no further calls. Model and prompt both changed, no single-cause claim. These are validation-development windows, not generalization evidence. See reports/direction_labels_f1_20261019.md.

## 2026-10-19 — Server8192 also truncated

Server connectivity verified with HTTP200/OK using existing system CA bundle. Real ETTh1 first Flash call exhausted8192 reasoning tokens with empty final content. Stop1/4; no valid F1. See reports/direction_label_server_20261019.md. Do not interpret raw failure-inclusive scoring zeros as measured predictive accuracy; confirm gateway thinking controls before further calls.

## 2026-10-19 — DAO ETTh1 label smoke response truncated

Run: runs/20261019_dao_etth1_labelsmoke. First Flash call returned length with empty final content; 2048 completion tokens were all reasoning tokens. Input43533 tokens. Stopped after1/4 attempts, no automatic retries. No valid F1 evidence. Preserve raw response privately. Next protocol/output-budget revision needs confirmation; do not convert failure into class0 or silently increase budget.
