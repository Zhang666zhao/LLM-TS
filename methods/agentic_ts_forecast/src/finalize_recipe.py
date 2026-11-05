"""Freeze selected recipe, generate history-only predictions, then seal single test score."""
from common import *
import sys,csv,importlib.metadata,importlib.util
from inference_recipe import Replay
name=sys.argv[1];mode=sys.argv[2]
if mode=='freeze':
 guard(600);assert not (RUN/'selection_frozen.json').exists(),'Selection already frozen'
 checkdir=RUN/'replay_checks'/name;checkpath=checkdir/'fullcheck.json' if (checkdir/'fullcheck.json').exists() else checkdir/'check.json';check=json.loads(checkpath.read_text());assert check['max_abs_delta']<check['tolerance']
 if checkpath.name=='fullcheck.json':assert check['validation_windows']==2817 and check['elements_over_tolerance']==0
 assert check.get('fresh_cache') is True and check['code_revision']==git_head(),'Final freeze requires fresh full validation replay under final code revision'
 files={};visited=set();invalid=set(json.loads((RUN/'invalid_candidates.json').read_text()).get('candidates',[])) if (RUN/'invalid_candidates.json').exists() else set()
 assert name not in invalid,'Cannot freeze quarantined candidate'
 def record(path):files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
 def visit(n):
  if n in visited:return
  assert n not in invalid,'Quarantined prediction dependency: '+n
  visited.add(n);d=ROOT/'runs'/n;record(d/'config.json');c=json.loads((d/'config.json').read_text())
  if c.get('compiled_from_config'):
   source=pathlib.Path(c['compiled_from_config']);assert hashlib.sha256(source.read_bytes()).hexdigest()==c['compiled_from_config_sha256'];record(source)
  if c.get('external_model_source'):
   source=pathlib.Path(c['external_model_source']);assert hashlib.sha256(source.read_bytes()).hexdigest()==c['external_model_source_sha256'];record(source)
  if c.get('training_memory'):
   artifact=pathlib.Path(c['training_memory']);assert hashlib.sha256(artifact.read_bytes()).hexdigest()==c['training_memory_sha256'];record(artifact)
   if artifact.with_suffix('.json').exists():record(artifact.with_suffix('.json'))
  for source in list(c.get('sources',{}))+c.get('source_ids',[]):
   if source.startswith('run:'):visit(source.split(':',1)[1])
  if c.get('engine')=='reloaded_finetune_v1':
   for p in (ROOT/'runs'/c['checkpoint_parent']/'trainer/finetuned-ckpt').glob('*'):
    if p.is_file():record(p)
  if c.get('engine') in ('quantile_average_v1','gated_semantics_v1','dense_memory_strength_v1'):visit(c['parent'])
  for p in d.glob('**/*.safetensors'):record(p)
  for p in d.glob('trainer/finetuned-ckpt/*.json'):record(p)
  if (d/'best.pth').exists():record(d/'best.pth')
  if (d/'weights.npy').exists():record(d/'weights.npy')
  if c.get('selection_pool'):
   artifact=pathlib.Path(c['selection_pool']);assert hashlib.sha256(artifact.read_bytes()).hexdigest()==c['selection_pool_sha256'];record(artifact)
 visit(name)
 for p in (RUN/'models').glob('**/*.safetensors'):record(p)
 for p in (RUN/'models').glob('**/*.json'):record(p)
 for pattern in ('**/*.pth','**/*.py'):
  for p in (RUN/'models').glob(pattern):record(p)
 record(pathlib.Path(CONTRACT['baseline_directory'])/'best.pth')
 record(pathlib.Path(CONTRACT['source']).parents[2]/'third_party/LTSF-Linear/models/DLinear.py')
 record(pathlib.Path(CONTRACT['source']))
 for f in ('dense_chronos_train.npz','dense_timesfm_train.npz'):
  if (RUN/'cache'/f).exists():record(RUN/'cache'/f)
 for package in ('chronos','timesfm3'):
  spec=importlib.util.find_spec(package)
  if spec and spec.submodule_search_locations:
   for directory in spec.submodule_search_locations:
    for source in pathlib.Path(directory).rglob('*.py'):record(source)
 record(ROOT/'benchmark/contract.json');record(ROOT/'src/official_eval.py');record(ROOT/'src/evaluate_frozen_pair.py')
 versions={}
 for package in ('torch','numpy','transformers','chronos-forecasting','timesfm3'):
  try:versions[package]=importlib.metadata.version(package)
  except importlib.metadata.PackageNotFoundError:versions[package]='metadata unavailable'
 frozen={'candidate':name,'frozen_epoch':time.time(),'deadline_epoch':budget()['deadline_epoch'],'code_revision':git_head(),'files':files,'runtime_versions':versions,'python':sys.version,'replay_check':check,'selection':'minimum observed full validation MSE; repeated adaptive validation, not an unbiased estimate','test_used_for_selection':False};save_json(RUN/'selection_frozen.json',frozen);print('SELECTION_FROZEN',name,flush=True)
elif mode=='predict':
 frozen=json.loads((RUN/'selection_frozen.json').read_text());assert frozen['candidate']==name;assert not (RUN/'sealed_test_result.json').exists()
 for path,sha in frozen['files'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,('Frozen dependency changed',path)
 assert git_head()==frozen['code_revision'],'Code changed after freeze'
 # Only history up to the last rolling origin is read here; no test labels constructed.
 with open(CONTRACT['source']) as f:
  reader=csv.DictReader(f);rows=[next(reader) for _ in range(14336)]
 raw=np.array([[float(r[k]) for k in COLS] for r in rows]);z=((raw-CONTRACT['scaler_mean'])/CONTRACT['scaler_scale']).astype(np.float32);origins=np.arange(11520,14337)
 replay=Replay(z,[r['date'] for r in rows],origins,RUN/'frozen_inference');p=replay.source('run:'+name);np.save(RUN/'test_predictions.npy',p)
 # Baseline is fixed in the benchmark, independently generated for a same-window comparison.
 baseline=replay.source('cache:dlinear');np.save(RUN/'test_baseline_predictions.npy',baseline)
 save_json(RUN/'frozen.json',dict(frozen,prediction_sha256=hashlib.sha256((RUN/'test_predictions.npy').read_bytes()).hexdigest(),baseline_prediction_sha256=hashlib.sha256((RUN/'test_baseline_predictions.npy').read_bytes()).hexdigest(),selection_frozen_sha256=hashlib.sha256((RUN/'selection_frozen.json').read_bytes()).hexdigest(),predictions_completed_epoch=time.time()));print('PREDICTIONS_READY',p.shape,flush=True)
else:raise ValueError(mode)
