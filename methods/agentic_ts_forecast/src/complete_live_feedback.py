"""Materialize missing training predictions for an unchanged numeric incumbent."""
from common import *
import sys
from live_feedback_agent import numeric
from observed_feedback import TOOLS,PARENT
name='20261105_livefeedback_complete_signal10_mse';old='20261105_livefeedback_signal10_val_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);z=np.load(RUN/'cache/trainval_z.npy');manifest=json.loads((RUN/'cache/window_manifest.json').read_text());results={}
for split,ts in manifest.items():
 guard(600);step=24 if split=='val' else 1;prior=np.load(ROOT/'runs'/PARENT/f'{split}_predictions.npy');past=np.stack([np.load(RUN/'strong_context_cache'/split/(n+'.npy')) for n in TOOLS],1);cor=np.load(RUN/'cache'/f'{split}_strong_observed_correction.npy');w=np.empty((len(ts),7))
 for i in range(0,len(ts),step):w[i:i+step]=numeric(z[ts[i]-512:ts[i]],past[i],'signal10')
 p=(prior+w[:,None]*cor).astype(np.float32)
 if split=='val':assert np.array_equal(p,np.load(ROOT/'runs'/old/'val_predictions.npy')),'Completing training artifacts changed validation'
 results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'));np.save(out/f'{split}_predictions.npy',p);np.save(out/f'{split}_strength.npy',w)
save_json(out/'config.json',{'engine':'live_feedback_agent_v1','mode':'signal10','part':'all','runtime_llm':False,'cadence':24,'source_ids':['run:'+PARENT]+['run:'+n for n in TOOLS],'derived_from':old,'validation_predictions_bitwise_identical':True,'training_queries_separated64h':'each gets fresh decision','code_revision':git_head()});save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated','new_hypothesis':False});print(name,{s:m['mse'] for s,m in results.items()},flush=True)
