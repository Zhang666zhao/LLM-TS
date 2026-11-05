"""Execute each future candidate from immutable tracked-source snapshot."""
from common import *
import sys
program=sys.argv[1]
if not program.endswith('.py') or '/' in program or not (ROOT/'src'/program).is_file():raise ValueError('Known src program required')
guard(600)
if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise RuntimeError('Dirty project; commit authorized source first')
rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();snap=RUN/'code_snapshots'/rev
for line in subprocess.check_output(['git','ls-tree','-r',rev,'src','benchmark'],cwd=ROOT,text=True).splitlines():
 meta,rel=line.split('\t',1);expected=meta.split()[2];data=(ROOT/rel).read_bytes();actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
 if actual!=expected:raise RuntimeError('Tracked content differs from commit despite stat cache: '+rel)
paths=subprocess.check_output(['git','ls-tree','-r','--name-only',rev,'src'],cwd=ROOT,text=True).splitlines()
for rel in paths:
 p=snap/rel
 if not p.exists():p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(subprocess.check_output(['git','show',rev+':'+rel],cwd=ROOT))
env=os.environ.copy();env['ETTH1_PROJECT_ROOT']=str(ROOT);env['ETTH1_CODE_REVISION']=rev
with (RUN/'snapshot_launches.jsonl').open('a') as f:f.write(json.dumps({'epoch':time.time(),'program':program,'args':sys.argv[2:],'revision':rev,'snapshot':str(snap),'pid':os.getpid()})+'\n')
os.execve(sys.executable,[sys.executable,'-u',str(snap/'src'/program),*sys.argv[2:]],env)
