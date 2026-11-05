"""Three declared point extraction combinations; frozen checkpoints and policy rules."""
from common import *
from inference_recipe import Replay
from transformed_tool import extract_point
from specialist_policy import run
import sys,copy
name='20261105_fullpoint_calendar1000_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True)
subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(1800)
parent='20261105_chronoscalft_duration1000_mse';levels=json.loads((RUN/'models/chronos2_29ec3766/config.json').read_text())['chronos_config']['quantiles']
cfg={'engine':'calendar_fullpoint_v1','checkpoint_parent':parent,'source_ids':['run:'+parent],'point_quantile_levels':levels,'point_rule':'full_integral','calendar':'hour','point_caveat':'CDF trapezoid integral with flat endpoint tails; no interval calibration claim','code_revision':git_head()};save_json(out/'config.json',cfg)
z=np.load(RUN/'cache/trainval_z.npy');dates=json.loads((RUN/'cache/dates.json').read_text());mf=json.loads((RUN/'cache/window_manifest.json').read_text());results={}
for split,ts in mf.items():
 replay=Replay(z,dates,ts,RUN/'calendar_fullpoint_cache'/split);q=replay.quantiles('chronos',model=parent,calendar='hour',levels=levels);p=extract_point(q,levels,'full_integral').astype(np.float32);np.save(out/f'{split}_predictions.npy',p);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'));save_json(out/'partial_metrics.json',results)
save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated'});print(name,{s:m['mse'] for s,m in results.items()},flush=True)
original=json.loads((ROOT/'runs/20261105_specialist_pointswap_core_mse/config.json').read_text());base={k:original[k] for k in ('source_ids','base_weights','variable_weights','rules')};base['source_ids'][2]='run:20261105_fullpoint_lora_mse'
for tag,calendar in [('core_lora',False),('core_lora_calendar',True)]:
 c=copy.deepcopy(base)
 if calendar:c['source_ids'][3]='run:'+name
 candidate=f'20261105_specialist_combo_{tag}_mse';run(c,candidate);save_json(ROOT/'runs'/candidate/'ablation_manifest.json',{'reference':'20261105_specialist_pointswap_core_mse','lora_full_integral':True,'calendar_full_integral':calendar,'weights_and_rules_unchanged':True,'code_revision':git_head()})
