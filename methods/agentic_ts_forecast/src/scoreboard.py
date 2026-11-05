"""Same-split evidence packet; never read test results during research."""
from common import *
rows=[]
invalid=set(json.loads((RUN/'invalid_candidates.json').read_text()).get('candidates',[])) if (RUN/'invalid_candidates.json').exists() else set()
for p in (ROOT/'runs').glob('20261105_*/metrics.json'):
 try:
  if p.parent.name in invalid:continue
  a=json.loads(p.read_text());m=a.get('metrics',a)
  if 'val' not in m or 'mse' not in m['val']:continue
  v=m['val']
  if v.get('windows')!=2817:continue
  cp=p.parent/'config.json';cfg=json.loads(cp.read_text()) if cp.exists() else {};runtime_llm=cfg.get('runtime_llm',any(s in p.parent.name for s in ('hybrid_','constrained_','reasonlow_','mempolicy_','strongagent_','selfagent_')))
  rows.append({'candidate':p.parent.name,'split':'validation','mse':v['mse'],'mae':v['mae'],'windows':v['windows'],'per_variable_mse':v['per_variable_mse'],'runtime_llm':runtime_llm,'elapsed_seconds':a.get('elapsed_seconds'),'metadata_path':str(p)})
 except (KeyError,ValueError,TypeError):continue
rows.sort(key=lambda a:a['mse']);save_json(RUN/'validation_leaderboard.json',rows)
packet={'metric':'float64 standardized all7 all64 all2817 MSE','split':'validation only, adaptively selected; not independent test','baseline':.5520404025597623,'best':rows[:10],'runtime_llm_best':[r for r in rows if r['runtime_llm']][:5],'candidates_full_validation':len(rows),'updated_epoch':time.time(),'deadline_epoch':budget()['deadline_epoch'],'test_scored':False}
save_json(RUN/'evidence_packet.json',packet)
print(json.dumps({'candidates':len(rows),'best':[(r['candidate'],r['mse']) for r in rows[:5]],'runtime_llm_best':[(r['candidate'],r['mse']) for r in packet['runtime_llm_best'][:2]]}),flush=True)
