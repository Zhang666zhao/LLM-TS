"""Memory-only API credential broker, bounded to the single authorized run."""
import os,sys,json,time,socketserver,threading,urllib.request,urllib.error,hashlib,pathlib,ssl
ROOT=pathlib.Path(__file__).resolve().parent.parent
RUN=ROOT/'runs/20261105_agentic_continuous_mse';B=json.loads((RUN/'budget.json').read_text())
SOCK=pathlib.Path('/tmp/etth1_20261105_'+str(os.getuid())+'.sock');CACHE=RUN/'api_cache';CACHE.mkdir(exist_ok=True)
lock=threading.Lock();sem=threading.BoundedSemaphore(B['api_concurrency_max']);count=json.loads((RUN/"api_usage.json").read_text()).get("requests_started",0) if (RUN/"api_usage.json").exists() else 0
key=sys.stdin.readline().strip()
if not key:raise RuntimeError('Missing API credential')
if SOCK.exists():raise RuntimeError('Existing broker socket; inspect process, do not duplicate')
if '--daemon' in sys.argv:
 pid=os.fork()
 if pid:print(json.dumps({'broker_pid':pid,'credential':'memory_only'}));sys.exit(0)
 os.setsid();log=os.open(str(RUN/'api_service.log'),os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
 os.dup2(log,1);os.dup2(log,2);os.dup2(os.open('/dev/null',os.O_RDONLY),0)
(RUN/'api_service.pid').write_text(str(os.getpid()))
def invoke(payload):
 global count
 if time.time()>=B['deadline_epoch']:raise RuntimeError('Budget expired')
 h=hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest();f=CACHE/(h+'.json')
 if f.exists():return json.loads(f.read_text())
 with sem:
  with lock:
   if count>=B['api_request_limit']:raise RuntimeError('Request budget exhausted')
   count+=1
   (RUN/'api_usage.json').write_text(json.dumps({'requests_started':count,'updated_epoch':time.time()}))
  start=time.time();out={'request_sha256':h,'started_epoch':start}
  try:
   req=urllib.request.Request('https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
   with urllib.request.urlopen(req,context=ssl.create_default_context(cafile="/etc/ssl/certs/ca-certificates.crt"),timeout=min(180,max(1,B['deadline_epoch']-time.time()))) as r:ans=json.load(r)
   ch=ans['choices'][0];out.update(status='ok',content=ch['message'].get('content',''),finish_reason=ch.get('finish_reason'),usage=ans.get('usage'),model=ans.get('model'),id=ans.get('id'))
  except urllib.error.HTTPError as e:out.update(status='http_error',http_status=e.code)
  except Exception as e:out.update(status='error',error_type=type(e).__name__)
  out['elapsed_seconds']=time.time()-start
  f.write_text(json.dumps(out,ensure_ascii=False));return out
class Handler(socketserver.StreamRequestHandler):
 def handle(self):
  try:out=invoke(json.loads(self.rfile.readline()))
  except Exception as e:out={'status':'broker_error','error':str(e).replace(key,'[REDACTED]')}
  self.wfile.write((json.dumps(out,ensure_ascii=False)+'\n').encode())
class Server(socketserver.ThreadingUnixStreamServer):daemon_threads=True
with Server(str(SOCK),Handler) as server:
 os.chmod(SOCK,0o600);server.timeout=1
 while time.time()<B['deadline_epoch']:server.handle_request()
SOCK.unlink(missing_ok=True)
