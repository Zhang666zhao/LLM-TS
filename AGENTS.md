# Direction-conditioned forecasting research

## Scope (2026-10-19)

- ORACLE NOISE v2 COMPLETE2026-10-19 23:12:16 CST:80/80 new evaluations,
  20 reused100percent oracle;100 results independently recomputed.175.625s
  after restart,706.822s since originalstart; originaldeadline preserved.
  Exact2817 targets/windows,noise ranks/offsets/actual counts/nested masks,
  oldcheckpoint hashes and unchanged modelstates verified;259 lightweight
  files synced/hash-matched. All100percent means match priorOracle exactly.
  MeanMSE accuracy60/70/80/90/100:
  A-L1 .409908/.392253/.373970/.356712/.339865;
  A-L2 .412422/.395734/.378157/.361099/.344318;
  B-L1 .413630/.395837/.377355/.359936/.342688;
  B-L2 .398707/.387401/.375495/.363873/.352629.
  Baseline mean.368701;90percent better and80percent worse in allfour.
  Single symmetric-noise seed,notLLM error structure or deployment threshold.
  Report reports/etth1_oracle_noise_20261019.md. v1 failure preserved, user
  authorized fix/restart succeeded. No training/API. Heartbeat outcome PAUSED.

- User explicitly authorizes fixing and restarting noise suite after v1
  preparation failure. Continue fixed80 evaluations with original deadline
  1792429229.48492, no budget reset. v1 evidence untouched; newv2 run directory.
  Import fix and actual-source check passed; launcher now inherits original
  budget.json via --budget-source. No extra methods/seeds/training/API.
  v2 LAUNCHED2026-10-19 23:09:20 CST, PID4117880, code5fcbe20;
  runs/20261019_etth1_oracle_noise_v2. budget.start1792422029.48492 and
  deadline1792429229.48492 preserved exactly; restart_at1792422560.6816704.
  Heartbeat outcome ACTIVE forv2 only; inspect completion, do not assume done.

- NOISE SUITE v1 STOPPED during preparation, before any GPU evaluation:
  AttributeError train.labels (function belongs to dataset module). PID4117517
  exited;0/80 evaluations ran. Preserve runs/20261019_etth1_oracle_noise_v1
  status/budget and original code1af0d71. Originaldeadline1792429229.48492
  remains unchanged. Fixed import in737ddfb and added read-only actual-source
  integration check; corrected server code research/oracle_noise_v2_prepared
  is NOT launched. Heartbeat outcome PAUSED. No automatic restart authorized;
  ask user to continue; do not turn a new directory into a budget reset.

- NEW authorized ETTh1 noisy-oracle inference sweep: label correctness
  60/70/80/90/100 percent, meaning20percent wrong at80percent correct.
  Five models x A/B x L1/L2 x four nonperfect levels=80 new evaluations;
  reuse20 old100percent oracle results. No training/API calls or label reuse32.
  Per-variable exact rounded corrupt counts, nested masks and uniform WRONG
  class replacement; noise seed20261019 shared across models/A-B/L1-L2.
  Original2817 test windows, same old oracle checkpoints and numerical scaler.
  Plan research/direction_conditioning/noise_sensitivity/PLAN.md. TwoA6000,
  new2h safety ceiling,13GiB free perGPU,fail-stop,no retries/tuning/extra seeds.
  LAUNCHED2026-10-19 23:00:29 CST, PID4117517, code1af0d71;
  runs/20261019_etth1_oracle_noise_v1, deadline1792429229.48492
  (2026-10-20 01:00:29 CST). Generic preflight and5 noise unit tests passed.
  Initial status preparing; completion NOT yet confirmed. Heartbeat outcome
  ACTIVE every10min, monitors only this80-evaluation suite. No auto-relaunch.

- DOWNSTREAM COMPLETE2026-10-19 13:21:28 CST;40/40 new jobs completed in
  1143.380s (19.06min), no retries/API/tuning. G1 20 inference +G2 20 training;
  reused25 core historical results and10 zero controls. Independent audit
  recomputed75 artifacts; all2817 test targets/windows and predicted labels
  exact;20 initial hashes and506 common-epoch order pairs match;10 B backbones
  equal old baseline tensors/buffers.379 lightweight files synced/hash-verified.
  G1 MSE worse thanG4 in20/20;G2 better thanG1 in18/20, better thanG4 in6/20,
  better thanmatched-zero in7/20. No stable label benefit established.
  Mean per-method G2 vsG4 MSE gains A-L1/A-L2/B-L1/B-L2:
  -4.249/-4.249/-0.144/-0.565%; vszero -0.034/-0.042/-0.091/-0.511%.
  DLinear-A zero already poor(.426912 vsbaseline.353263); do not attribute
  its entire G2 gap to LLM error. OT can worsen despite all-variable gains.
  Report reports/etth1_llm_downstream_20261019.md. CPU2021/CUDA2022 disclosed,
  single-seed/stride32 correlation, G3 oracle not deployable. No processes left.
  Heartbeat outcome PAUSED after completion audit; no further runs authorized.

- NEW authorization: execute ETTh1 downstream four-group comparison on two
  server A6000 GPUs. Fixed G1 20 old-oracle-checkpoint LLM-label evaluations;
  G2 20 LLM-condition train/validation/test jobs. Reuse G3 20 oracle, G4 five
  baselines, ten zero controls. No new API, other datasets, tuning or retries.
  Canonical plan research/direction_conditioning/downstream/PLAN.md. Preserve
  original 7-variable 512/64 data/scaler/splits and CPU2021/CUDA2022. New safety
  ceiling10h includes preparation; do not reuse/reset any prior deadline.
  A fresh joint initialization; B frozen old base_zero. Labels from completed
  allsplits NPZ only, stride32 causal reuse. Fail-stop and exact paired checks.
  LAUNCHED2026-10-19 13:02:24 CST, PID4076760, code20fdc14;
  run runs/20261019_etth1_llm_downstream_v1, deadline1792422144.9447007
  (23:02:24 CST). Generic preflight and3 metric unit tests passed. Labels/data
  and35 historical references verified; all five GPU contracts passed. G1
  DLinear A-L1/A-L2 dispatched onGPU0/1; NOT yet completed training.
  Heartbeat outcome ACTIVE every10min for this suite only.
  Inspect status/completion; never relaunch or reset budget automatically.

- ALL-SPLIT API BATCH COMPLETE: runs/20261019_dao_etth1_allsplits_v1 finished
  2026-10-19 12:36:27 CST, elapsed3820.342s; verified independently at12:44+.
  All431 anchors valid (train253/val89/test89),13699 windows/95893 unique
  window-variable rows.346 new calls=342 unique+4 same-request joint-conflict
  repairs;no429;all reasoning present.8,453,107 new tokens. No budget reset.
  MacroF1/accuracy L1,L2: train .358922/.646958,.559124/.630307;
  val .403547/.632639,.604884/.635022;test .351687/.683199,.475805/.620062.
  Both test F1 below0.5;no test-driven tuning or reruns. Verified full raw truth,
  CSV/JSONL uniqueness,NPZ labels/index alignment,causal stride32,prefix input
  normalization,all6 aggregate+42 per-variable sklearn metrics,exact old val
  predictions and source hashes.16 result artifacts copied locally,hash matched.
  Report reports/direction_labels_allsplits_20261019.md; local runs/... contains
  CSV/JSONL,labels/{train,val,test}.npz,metrics and verification.json.
  Heartbeat outcome PAUSED. Label coverage no longer blocks strict downstream
  train/val/test comparison; NO downstream model training or extra API calls
  launched/authorized by completion. Await the next execution request.

- ALL-SPLIT API BATCH LAUNCHED on GPU_Foundation:
  runs/20261019_dao_etth1_allsplits_v1, PID4075112. Original deadline
  1792387967.1172173 (2026-10-19 13:32:47 CST), never reset. Preparation verified
  all431 anchors and13699 windows;89 val cached exactly;253 train+89 test pending.
  Causal-prefix scaler invariance/epsilon checks passed. Local27 tests and
  training-autoresearch preflight passed. Heartbeat outcome ACTIVE, repurposed
  exclusively to this API batch every10min. Launch confirmed; completion NOT
  yet confirmed. Check status/completion before any next operation.

- Latest user authorizes completing LLM labels for ORIGINAL ETTh1 train/val/test.
  Reuse all89 validation anchors, add253 train+89 test=342 unique requests;
  stride32 causal first-window carry, same Pro/concise prompt/reasoningON,
  context512/horizon64, seven variables and both L1/L2. No downstream training.
  New all_splits_server.py snapshots code and separates inference from scoring.
  Fixed safety envelope:2h including prep,30 max in-flight,10s admission,
  <=410 attempts total, <=3 attempts/anchor; bounded identical-request recovery
  for transient/429/schema/length/joint conflicts only, never repeat a success.
  Auth/contract/nonfinite errors stop; no automatic budget extension or tuning.
  Input scaler fits raw prefix[0:min(anchor_target,8640)] to prevent future
  training statistics entering early training prompts. Val/test preprocessing
  identical to prior val; require offline label-boundary equality and inactive
  epsilon floors. Old oracle label definition remains unchanged.
  Expected13699 windows/95893 unique window-variable rows,431 valid anchors.
  Record actual launch and completion separately; this entry alone is not launch.

- Latest request: PLAN ONLY for ETTh1 downstream comparison using the completed
  stride32 LLM-label cache, zero new API calls. Plan is
  research/direction_conditioning/DOWNSTREAM_PLAN.md. Four groups: G1 existing
  oracle checkpoints with predicted labels; G2 same A/B training recipe using
  predicted conditioning labels (numeric targets remain true future values);
  G3 existing oracle; G4 existing five baselines. Preserve L1/L2 × A/B separately.
  Server audit confirmed all35 ETTh1 checkpoints available. Cached LLM labels
  cover validation target starts8640..11456 ONLY, not original train512..8576
  or test11520..14336. Old main metrics are TEST; use best_validation metrics or
  frozen-checkpoint validation replay for same-window comparisons. G2 strict
  retraining lacks original training LLM labels and cannot run under current
  no-API scope. A chronological local-training alternative is documented but
  NOT selected/authorized: it changes the benchmark and cannot isolate label
  source effects. No training/inference/API job launched for this plan.

- Stride32 COMPLETE2026-10-19: finalrun20261019_dao_etth1_stride32_v5,89/89 anchors valid,2817 windows,19719 unique window-variable rows(allL1/L2 present). Causal first-window carry0..31 verified; all aggregate and14 per-variable F1 verified sklearn. Full carried L1 F1 .4035474363,accuracy .6326385719;L2 F1 .6048844426,accuracy .6350220599. Anchor-only F1 .4687636905/.6440830913. True-label carry agreement .8009026827/.8852375881.89 new calls=85 unique+4 repairs(two429,twojoint conflicts),plus4 oldcache,984.306s. Complete CSV/JSONL/metrics copied locally to runs/...stride32_v5; report reports/direction_labels_stride32_20261019.md. STOP: no further API jobs, no monitors active, no training. All prior runs/failed requests preserved; old fullval prep never launched.

- Latest stride32 continuation v4: v3 encountered anotherHTTP429 after36 requests, drained completely;74/89 anchors valid. v4 reuses74, sends15 at10s spacing,max30 in-flight,same original deadline1792380350.1771271/model/prompt/labels. Global attempted prior73 +15=88 (cap89), all failures preserved; no successful repeats. Actualrun20261019_dao_etth1_stride32_v4, inspect completion/metrics before further API calls. Final19719-row CSV/JSONL generated by scorer.

- Stride32 active continuation v3: v2 sent37, retained35 valid new+4 old=39 anchors; oneHTTP429, one joint-label inconsistency. All in-flight drained and v2 preserved. v3 reuses39, sends50 pending/repair requests at3.5s admission spacing, max30 in-flight, same Pro/prompt/labels and original deadline1792380350.1771271. Recorded one bounded repair pass, no successful requests repeated. Combined new calls37+50=87 (cap89). Actual run20261019_dao_etth1_stride32_v3, PID4073963; completion.json/metrics+CSV/JSONL will be produced automatically. Inspect before further calls.

- Latest user RESUMES with stride32 label reuse: new run20261019_dao_etth1_stride32_v1,89 causal first-window anchors covering2817 validation windows,30 concurrency,Pro concise prompt/thinkingON,16384max output,30min safety deadline,one request per uncached anchor,no automatic retries. Score all19719 window-variable rows against own true labels; retain anchor ID/age and per-label/per-variable metrics. Old partial fullval preparation remains preserved and unlaunched.
- Actual launch uses stride32_v2: v1 cache preflight found pilot manifest default-model mismatch before any API dispatch. Fixed to check effective request model override; preserve v1. v2 verified4 matching cached anchors,85 new requests,89 total. API source/prompt/threshold unchanged. Run started; inspect status/completion before relaunching.

- User PAUSED full-validation work before new API dispatch. Fullval preparation process4073316 terminated on request; no launch.json or API calls in that run. Preserve partial inputs in runs/20261019_dao_etth1_fullval_v1, do not resume or launch it automatically. Latest proposed simplification: reuse one per-variable L1/L2 vector for each32 consecutive windows. Discuss causal anchor definition and scoring before a new version; pause remains in force.

- Latest user instruction: start next experiment at30 concurrency. New round: frozen Pro concise prompt,30 equispaced ETTh1 validation windows,target starts8704..11392 (exclude pilot targets),30 requests,16384 output tokens each,30min ceiling incl preparation,no retries,server-only,zero GPU training. Run20261019_dao_etth1_c30_v1. Record completion after checking; do not relaunch existing run.
- C30 COMPLETE verified:30/30 responses valid,all reasoning>0,all stop.175.533s,dispatch0.361s.210 samples per label: L1 macroF1 .4719079875,accuracy142/210;L2 .6863887565,accuracy147/210.Independent sklearn aggregate and per-variable check passed.30 requests/no retries;726540 tokens. Report reports/direction_labels_c30_20261019.md. No further jobs launched; remaining budget not rolled forward.

- User authorizes concurrency up to50. Implemented concurrent_server.py with bounded admission, explicit total request/time budgets, default dry-run and no retries. No new windows/API runs are authorized merely by raising concurrency. Keep reasoning enabled, server-only. Gateway50-way capacity not yet live-tested.

- Latest bounded exploration COMPLETE: server run20261019_dao_compact_f1/concise_v1, first complete version pro-concise (aliyun/DeepSeek-v4-pro), reasoning enabled. Two fixed val windows x7 variables, 28 labels valid. L1 Macro-F1 .242424, accuracy8/14 (all0); L2 Macro-F1 .3000, accuracy6/14. Below0.5. sklearn independently confirmed.8 attempted API requests total incl interrupted2,971s; stop, no more automatic runs. Report reports/direction_labels_f1_20261019.md. Earlier Flash partials not mixed into Pro metrics.

- Latest authorization supersedes earlier fail-stop-only scopes: server-only bounded protocol exploration, max12 API requests/30min, max16384 output tokens/request, reasoning ON, zero GPUs/training. Fixed2 ETTh1 validation windows x7 variables. Select first complete valid prompt version without reading F1; score all versions afterwards. Run20261019_dao_compact_f1, poll during active turn; stop on auth/network failures or budget. Preserve previous runs.
- Dedicated branch: research/direction-conditioned-forecasting; separate worktree.
- Current authorization: first ETTh1 validation label screening, two fixed windows x two DAO models (four requests maximum), no retries/tuning/downstream training. Goal Macro-F1 >0.5 is aspirational, not a stop-until-achieved loop. Secret must be supplied securely through environment.
- User requires API execution on GPU_Foundation, never local Mac. Server connectivity confirmed HTTP200/OK on2026-10-19. Existing system CA bundle /etc/ssl/certs/ca-certificates.crt resolves Conda trust failure without disabling TLS. New server-only run 20261019_dao_etth1_server8192 launched with output8192, same two windows/models, four-call/900s bound; inspect status before any further request. No GPU training.
- Server8192 run STOPPED: first request consumed8192 reasoning tokens, empty content, finish_reason=length; remaining3 not sent. No valid F1. Its raw metrics.json counts missing predictions as incorrect by design; zero entries are failure-inclusive bookkeeping, NOT measured label accuracy. Need confirmed gateway thinking control before any new run.
- 2026-10-19 screening stopped after first request: DAO Flash connected, 43533 prompt tokens and 2048 reasoning/output tokens, finish_reason=length, empty final content. No scored prediction, no F1 claim. Remaining three calls not sent. See reports/direction_label_smoke_20261019.md; do not automatically retry or increase output budget.
- Do not launch training, paid API calls, upload time-series data, or change environments without a separately specified scope and budget.
- Fixed interface: observed multivariate history -> per-variable labels in {-1,0,+1}.
- New label definitions and thresholds are configurable research choices. Historical oracle_v1 used tau=0.5, epsilon=0.001, ddof=0; NEVER rewrite those results as another threshold.
- No future values in inference inputs, feature construction, retrieval selection, or prompt tuning. Training labels may use their own future targets.
- Full historical series is mandatory; historical summaries and verified semantics are optional additions, not replacements. Never invent semantics or causality.
- Select prompt/threshold on train/validation only; preserve chronological boundaries and evaluate frozen choices on test.
- Preserve A-L1, A-L2, B-L1, B-L2 and respective matched-zero controls separately.
- Historical CPU/backbone seed2021, actual CUDA2022; any correction is a NEW experiment version.
- Local historical source copies are not portable/reverified reproductions. Remote SSH authentication blocked fresh audit on2026-10-19; see provenance.
- No raw data, checkpoints, label caches, prediction arrays, credentials or environment folders in Git.
- Preserve existing benchmark code and unrelated edits; do not refactor frozen copies.
