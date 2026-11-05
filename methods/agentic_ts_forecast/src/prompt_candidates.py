"""First simple-to-rich direct forecasting candidates, same train-internal windows."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from common import *
import sys
META='ETTh1 hourly transformer measurements. HUFL/HULL: High Useful/Useless Load; MUFL/MULL: Middle Useful/Useless Load; LUFL/LULL: Low Useful/Useless Load; OT: oil temperature. Exact units and electrical topology unknown. All supplied values are train-standardized z scores; negatives are valid. Predict all seven channels. Do not force positive values. Daily/weekly patterns may vary by channel.'
def stats(x):
 d={}
 for n in [24,64,168,512]:
  a=x[-n:];d[str(n)]={'mean':a.mean(0).round(3).tolist(),'std':a.std(0).round(3).tolist(),'first_to_last_mean_change':(a[-min(n//2,24):].mean(0)-a[:min(n//2,24)].mean(0)).round(3).tolist()}
 for k in [24,168]:d['acf_'+str(k)]=[round(float(np.corrcoef(x[:-k,c],x[k:,c])[0,1]),3) if x[:,c].std()>1e-8 else 0 for c in range(7)]
 return d
def run_candidate(level):
 name=f'20261105_deepseek_prompt{level}_nothink_mse';out=ROOT/'runs'/name
 if (out/'metrics.json').exists():print('EXISTS',name,flush=True);return
 out.mkdir(exist_ok=True);(out/'requests').mkdir(exist_ok=True)
 subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True)
 z=np.load(RUN/'cache/trainval_z.npy');manifest=json.loads((RUN/'cache/window_manifest.json').read_text());allts=np.array(manifest['dev']);ts=allts[np.linspace(0,len(allts)-1,8,dtype=int)];x,y=windows(z,ts)
 config={'candidate':name,'level':level,'primary_change':['raw_history','author_metadata','historical_statistics'][level],'origins':ts.tolist(),'split':'training_internal_screening','test_access':False,'max_tokens':6000,'thinking':'disabled','code_revision':git_head(),'budget_file':str(RUN/'budget.json')};save_json(out/'config.json',config);(out/'git_commit.txt').write_text(git_head());(out/'data_version.txt').write_text((ROOT/'data_version.txt').read_text())
 start=time.time();pred=np.zeros_like(y);records=[]
 def one(i):
  guard(600);instruction='Forecast the next 64 hourly observations from the supplied 512 observations. Columns are '+','.join(COLS)+'. Return ONLY JSON {"predictions": [[7 numbers], ... exactly 64 rows]}. All numbers must be finite. Keep numerical precision to 3 decimals. No prose. Use historical patterns; do not hallucinate external observations.'
  if level>=1:instruction+='\n'+META
  obs={'history_512_by_7':np.round(x[i],3).tolist(),'horizon':64}
  if level>=2:obs['multiscale_statistics']=stats(x[i])
  messages=[{'role':'system','content':instruction},{'role':'user','content':json.dumps(obs,separators=(',',':'))}]
  save_json(out/'requests'/f'{i:03d}.json',messages)
  answer=api(messages,max_tokens=6000);save_json(out/'requests'/f'{i:03d}_response.json',answer)
  if answer['status']!='ok':return i,None,answer['status']
  try:
   a=np.asarray(parse_json(answer['content'])['predictions'],dtype=np.float32)
   if a.shape!=(64,7) or not np.isfinite(a).all():raise ValueError('shape or finite')
   return i,a,'ok'
  except Exception as e:return i,None,'invalid_'+type(e).__name__
 with ThreadPoolExecutor(max_workers=4) as pool:
  for f in as_completed([pool.submit(one,i) for i in range(len(ts))]):
   i,a,status=f.result();records.append({'index':i,'status':status});print(name,i,status,flush=True)
   if a is not None:pred[i]=a
 save_json(out/'response_status.json',records)
 failed=[r for r in records if r['status']!='ok']
 result={'candidate':name,'code_revision':git_head(),'elapsed_seconds':time.time()-start,'status':'failed' if failed else 'screened','failed_windows':len(failed),'split':'train_internal','origins':ts.tolist(),'not_official_champion':True}
 if not failed:result.update(metric(pred,y));np.save(out/'predictions.npy',pred);np.save(out/'targets.npy',y)
 save_json(out/'metrics.json',result)
 with (RUN/'experiment_index.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
 save_json(RUN/'status.json',{'phase':'simple_prompts','latest':name,'updated_epoch':time.time(),'experiments_completed':level+1});print(json.dumps(result),flush=True)
if __name__=='__main__':
 for level in range(3):run_candidate(level)
