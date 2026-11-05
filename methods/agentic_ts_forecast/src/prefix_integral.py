"""Re-evaluate existing prefix-fitted models with full CDF points; no model refitting."""
from common import *
from transformed_tool import predict
import sys

def run(cut):
 assert cut in (3072,4096,6272);levels=json.loads((RUN/'models/chronos2_29ec3766/config.json').read_text())['chronos_config']['quantiles'];z=np.load(RUN/'cache/trainval_z.npy')[:8640]
 for mode in ('semantic','pca','random'):
  parent=f'20261105_prefixview_{cut}_{mode}_mse';src=ROOT/'runs'/parent;name=f'20261105_prefixintegral_{cut}_{mode}_mse';out=ROOT/'runs'/name;assert not (out/'config.json').exists();out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(2400);cfg=json.loads((src/'config.json').read_text());cfg.update(point_rule='full_integral',point_quantile_levels=levels,checkpoint_parent=parent,point='21 quantile CDF trapezoid integral, flat endpoint tails; inverse POINT transform',source_ids=['run:'+parent],code_revision=git_head());ts=np.load(src/'origins.npy');y=np.load(src/'targets.npy');assert ts.min()-512>=cut and ts.max()+64<=cfg['labels_end']<=8640;save_json(out/'config.json',cfg);p=predict(z,ts,cfg);assert p.shape==y.shape and np.isfinite(p).all();old=np.load(src/'predictions.npy');np.save(out/'predictions.npy',p);np.save(out/'origins.npy',ts);np.save(out/'targets.npy',y);result={'candidate':name,'metrics':{'prefix':metric(p,y)},'matched_old_point_metrics':metric(old,y),'status':'evaluated','no_official_validation_or_test':True};save_json(out/'metrics.json',result);print(name,result['metrics']['prefix']['mse'],'old',result['matched_old_point_metrics']['mse'],flush=True)

def build():
 from view_memory_gate import features
 z=np.load(RUN/'cache/trainval_z.npy')[:8640];orig=[];fold=[];pred=[];truth=[];sources=[]
 for cut in (3072,4096,6272):
  pp=[];ts=None;y=None
  for mode in ('semantic','pca','random'):
   out=ROOT/'runs'/f'20261105_prefixintegral_{cut}_{mode}_mse';cfg=json.loads((out/'config.json').read_text());t=np.load(out/'origins.npy');yy=np.load(out/'targets.npy');a=np.load(out/'predictions.npy');assert cfg['cutoff']==cut and t.min()-512>=cut and t.max()+64<=cfg['labels_end']<=8640
   if ts is not None:assert np.array_equal(ts,t) and np.array_equal(y,yy)
   ts=t;y=yy;pp.append(a);sources.append({'name':out.name,'config_sha256':hashlib.sha256((out/'config.json').read_bytes()).hexdigest(),'prediction_sha256':hashlib.sha256((out/'predictions.npy').read_bytes()).hexdigest()})
  orig.extend(ts);fold.extend([cut]*len(ts));pred.append(np.stack(pp,1));truth.append(y)
 ts=np.array(orig);pred=np.concatenate(pred);truth=np.concatenate(truth);x,_=windows(z,ts);f=features(x,pred);path=RUN/'cache/view_prefix_integral_memory.npz';assert not path.exists();np.savez_compressed(path,origins=ts,fold=np.array(fold),predictions=pred,targets=truth,features=f);assert max(ts[np.array(fold)<6272])+64<=min(ts[np.array(fold)==6272])-512;save_json(path.with_suffix('.json'),{'case_count':len(ts),'sources':sources,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'caveats':['Same prefix-fitted checkpoints as old bank; only point extraction changed','Overlapping cases not independent; fulltraining scaler shared','Proxy fitted tools still differ from final fulltraining tools'],'code_revision':git_head()});print('INTEGRAL_MEMORY_BUILT',len(ts),f.shape,flush=True)
if __name__=='__main__':
 if sys.argv[1]=='build':build()
 else:run(int(sys.argv[1]))
