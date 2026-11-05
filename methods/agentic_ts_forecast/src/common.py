import os,json,time,pathlib,socket,subprocess,hashlib
import numpy as np
ROOT=pathlib.Path(os.environ.get('ETTH1_PROJECT_ROOT',str(pathlib.Path(__file__).resolve().parent.parent)))
RUN=ROOT/'runs/20261105_agentic_continuous_mse'
CONTRACT=json.loads((ROOT/'benchmark/contract.json').read_text())
COLS=CONTRACT['variables']
def budget():return json.loads((RUN/'budget.json').read_text())
def guard(reserve=0):
 if time.time()+reserve>=budget()['deadline_epoch']:raise RuntimeError('Wall-clock budget reached')
def api(messages,max_tokens=8192,thinking="disabled",reasoning_effort=None):
 guard();payload={'model':'deepseek-v4-pro','messages':messages,'temperature':0,'max_tokens':max_tokens,'stream':False,'thinking':{'type':thinking}}
 if reasoning_effort is not None:payload['reasoning_effort']=reasoning_effort
 with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as s:
  s.settimeout(min(600,max(1,budget()['deadline_epoch']-time.time())));s.connect('/tmp/etth1_20261105_'+str(os.getuid())+'.sock');s.sendall((json.dumps(payload,ensure_ascii=False)+'\n').encode());f=s.makefile('rb');answer=json.loads(f.readline())
 if answer.get('status')=='http_error' and answer.get('http_status') in (401,403):
  save_json(RUN/'authorization_stop.json',{'epoch':time.time(),'http_status':answer['http_status']});raise SystemExit('API authorization failure: stop, no retries')
 return answer
def parse_json(text):
 text=text.strip()
 if text.startswith('```'):text='\n'.join(text.splitlines()[1:-1])
 a=text.find('{');b=text.rfind('}')
 if a>=0 and b>=a:return json.loads(text[a:b+1])
 return json.loads(text)
def metric(pred,y):
 p=np.asarray(pred);assert p.shape==y.shape and np.isfinite(p).all()
 e=p.astype('float64')-y.astype('float64')
 return {'mse':float(np.mean(e*e)),'mae':float(np.mean(np.abs(e))),'per_variable_mse':dict(zip(COLS,np.mean(e*e,axis=(0,1)).tolist())),'horizon_mse':np.mean(e*e,axis=(0,2)).tolist(),'windows':len(y)}
def git_head():return os.environ.get('ETTH1_CODE_REVISION') or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def save_json(path,obj):pathlib.Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
def windows(z,origins):return np.stack([z[t-512:t] for t in origins]),np.stack([z[t:t+64] for t in origins])
