"""Re-evaluate existing prefix-fitted models with full CDF points; no model refitting."""
from common import *
from transformed_tool import predict
import sys

def build(early=False):
 from view_memory_gate import features
 z=np.load(RUN/'cache/trainval_z.npy')[:8640];orig=[];fold=[];pred=[];truth=[];sources=[]
 for cut in ((1024,3072,4096,6272) if early else (3072,4096,6272)):
  pp=[];ts=None;y=None
  for mode in ('semantic','pca','random'):
   out=ROOT/'runs'/f'20261105_prefixcalendar_{cut}_{mode}_mse';cfg=json.loads((out/'config.json').read_text());t=np.load(out/'origins.npy');yy=np.load(out/'targets.npy');a=np.load(out/'predictions.npy');assert cfg['cutoff']==cut and t.min()-512>=cut and t.max()+64<=cfg['labels_end']<=8640
   if ts is not None:assert np.array_equal(ts,t) and np.array_equal(y,yy)
   ts=t;y=yy;pp.append(a);sources.append({'name':out.name,'config_sha256':hashlib.sha256((out/'config.json').read_bytes()).hexdigest(),'prediction_sha256':hashlib.sha256((out/'predictions.npy').read_bytes()).hexdigest()})
  orig.extend(ts);fold.extend([cut]*len(ts));pred.append(np.stack(pp,1));truth.append(y)
 ts=np.array(orig);pred=np.concatenate(pred);truth=np.concatenate(truth);x,_=windows(z,ts);f=features(x,pred);path=RUN/('cache/view_prefix_earlycalendar_memory.npz' if early else 'cache/view_prefix_calendar_memory.npz');assert not path.exists();np.savez_compressed(path,origins=ts,fold=np.array(fold),predictions=pred,targets=truth,features=f);assert max(ts[np.array(fold)<6272])+64<=min(ts[np.array(fold)==6272])-512;save_json(path.with_suffix('.json'),{'case_count':len(ts),'sources':sources,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'caveats':['New calendar-hour prefix-fitted checkpoints; same chronological queries and21quantile point','Overlapping cases not independent; fulltraining scaler shared','Proxy fitted tools still differ from final fulltraining tools'],'code_revision':git_head()});print('INTEGRAL_MEMORY_BUILT',len(ts),f.shape,flush=True)
if __name__=='__main__':build(len(sys.argv)>1 and sys.argv[1]=='early')
