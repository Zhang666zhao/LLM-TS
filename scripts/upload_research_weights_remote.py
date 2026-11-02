"""Upload one verified weight bundle from the source server to GitHub Release.

GitHub credential travels through encrypted SSH stdin only, never argv/files.
Archives contain only the already audited manifest's checkpoint files.
"""
import hashlib,json,shlex,subprocess,sys
from pathlib import Path
from package_research_weights import ROOT,OUT,RUNS

def main():
    dataset=sys.argv[1]
    rows=[r for r in json.loads((OUT/'weights_manifest.json').read_text())['weights'] if r['dataset']==dataset]
    if not rows:raise ValueError('Unknown dataset')
    name='weights-'+dataset+'-20261102.tar';local=OUT/name
    h=hashlib.sha256()
    with local.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    releases=json.loads(subprocess.check_output(['gh','api','repos/Zhang666zhao/LLM-TS/releases?per_page=100'],text=True))
    release=next(r for r in releases if r['tag_name']=='forecast-weights-20261102')
    assert release['draft']
    if any(a['name']==name for a in release['assets']):raise ValueError('Asset already exists; refusing overwrite')
    request=dict(rows=rows,name=name,sha256=h.hexdigest(),bytes=local.stat().st_size,release_id=release['id'],runs=RUNS)
    remote=r'''
import sys,json,tarfile,tempfile,hashlib,http.client,urllib.parse
from pathlib import Path
request=json.loads(sys.stdin.readline());token=sys.stdin.readline().strip()
assert token and request['name'].startswith('weights-')
with tempfile.TemporaryDirectory(prefix='llmts-weights-upload-') as directory:
 p=Path(directory)/request['name'];root=Path(request['runs'])
 with tarfile.open(p,'w') as archive:
  for r in request['rows']:
   source=root/r['checkpoint'];h=hashlib.sha256()
   with source.open('rb') as f:
    for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
   assert h.hexdigest()==r['sha256']
   archive.add(source,arcname=r['checkpoint'],recursive=False)
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 assert h.hexdigest()==request['sha256'] and p.stat().st_size==request['bytes']
 connection=http.client.HTTPSConnection('uploads.github.com',timeout=300)
 path='/repos/Zhang666zhao/LLM-TS/releases/'+str(request['release_id'])+'/assets?name='+urllib.parse.quote(request['name'])
 connection.putrequest('POST',path)
 for k,v in {'Authorization':'Bearer '+token,'User-Agent':'LLM-TS-Weight-Archive','Content-Type':'application/x-tar','Content-Length':str(p.stat().st_size),'Accept':'application/vnd.github+json'}.items():connection.putheader(k,v)
 connection.endheaders()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):connection.send(b)
 response=connection.getresponse();body=response.read();connection.close()
 if response.status!=201:raise SystemExit('GitHub upload HTTP '+str(response.status)+'; no retry')
 result=json.loads(body)
 assert result['size']==request['bytes'] and result['state']=='uploaded'
 assert result.get('digest')=='sha256:'+request['sha256']
 print(json.dumps({k:result.get(k) for k in ['name','size','state','digest']}))
'''
    token=subprocess.check_output(['gh','auth','token'],text=True).strip()
    result=subprocess.run(['ssh','-o','ClearAllForwardings=yes','-o','BatchMode=yes','-o','ConnectTimeout=10',
        'GPU_Foundation','python3 -c '+shlex.quote(remote)],input=json.dumps(request)+'\n'+token+'\n',text=True,capture_output=True)
    del token
    print(result.stdout)
    if result.returncode:raise SystemExit('Remote upload failed; inspect release assets before retry. '+result.stderr[-400:])

if __name__=='__main__':main()
