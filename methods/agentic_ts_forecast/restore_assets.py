"""Restore checksum-verified benchmark files from a release archive or existing copy."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import numpy as np

ROOT=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def install(stream,dest,expected):
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent,delete=False) as temp:
        tmp=Path(temp.name)
        try:
            shutil.copyfileobj(stream,temp)
            temp.flush()
            if sha(tmp)!=expected:raise ValueError(f'Checksum mismatch: {dest}')
            temp.close()
            os.replace(tmp,dest)
        finally:
            tmp.unlink(missing_ok=True)

def generate_local_inputs(group):
    if group in ('all','ETTh1'):
        contract=json.loads((ROOT/'benchmark/contract.json').read_text())
        for split in ('val','test'):
            start=contract['split'][split+'_start'];end=contract['split'][split+'_end']
            np.save(ROOT/'benchmark'/f'{split}_starts.npy',np.arange(start,end-512-64+1,dtype=np.int64))
    if group in ('all','ETTm1'):
        # Build a private example on the user's machine; only its preceding 512 observations are written.
        with (ROOT/'transfer/assets/ETTm1.csv').open() as f:
            r=csv.DictReader(f);rows=[next(r) for _ in range(46080)][-512:]
        with (ROOT/'transfer/artifacts/ETTm1/example_history.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=r.fieldnames);w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--archive',type=Path,help='Downloaded release tar.gz; reads only manifest-listed regular files')
    mode.add_argument('--from-package',type=Path,help='Existing fully provisioned agentic_ts_forecast directory')
    mode.add_argument('--source-root',help='Shared agentic_TS directory containing LLM_predict and TSRAG')
    ap.add_argument('--ssh',help='SSH host alias for a remote --source-root')
    ap.add_argument('--dataset',choices=['ETTh1','ETTm1','all'],default='all')
    a=ap.parse_args()
    if a.ssh and (not a.source_root or a.ssh.startswith('-')):ap.error('--ssh requires --source-root and a host alias')
    manifest=json.loads((ROOT/'restore_manifest.json').read_text())
    pending={}
    for rel,r in manifest.items():
        if a.dataset not in ('all',r['group']):continue
        dest=ROOT/rel
        if dest.exists():
            if sha(dest)!=r['sha256']:raise ValueError(f'Existing file differs; inspect or move {dest}')
        else:pending[rel]=r
    if a.archive:
        seen=set()
        with tarfile.open(a.archive,mode='r:gz') as t:
            for item in t:
                if not item.isfile() or item.name not in manifest or item.name in seen:
                    raise ValueError(f'Unexpected or duplicate archive member: {item.name}')
                seen.add(item.name)
                if item.name in pending:
                    r=pending[item.name]
                    if item.size!=r['bytes']:raise ValueError('Archive size mismatch')
                    install(t.extractfile(item),ROOT/item.name,r['sha256'])
        if set(pending)-seen:raise ValueError('Archive is missing required files')
    elif pending and a.ssh:
        remote={r['source']:(ROOT/rel,r['sha256']) for rel,r in pending.items()}
        cmd='tar -C '+shlex.quote(a.source_root)+' -cf - -- '+shlex.join(remote)
        with subprocess.Popen(['ssh','-o','BatchMode=yes','--',a.ssh,cmd],stdout=subprocess.PIPE) as p:
            with tarfile.open(fileobj=p.stdout,mode='r|') as t:
                for item in t:
                    if not item.isfile() or item.name not in remote:raise ValueError('Unexpected archive member')
                    dest,h=remote.pop(item.name);install(t.extractfile(item),dest,h)
            if p.wait() or remote:raise RuntimeError('Incomplete transfer; rerun after checking SSH and the shared root')
    else:
        for rel,r in pending.items():
            path=a.from_package/rel if a.from_package else Path(a.source_root)/r['source']
            with path.open('rb') as f:install(f,ROOT/rel,r['sha256'])
    generate_local_inputs(a.dataset)
    print(f'Assets verified for {a.dataset}; restored {len(pending)} files.')

if __name__=='__main__':main()
