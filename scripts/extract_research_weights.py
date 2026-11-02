"""Verify release archive and allowlisted checkpoint hashes before safe extraction."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

def digest(stream):
    h=hashlib.sha256()
    for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','bundles','manifest','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    bundle=next(r for r in json.loads(a.bundles.read_text()) if r['name']==a.bundle.name)
    with a.bundle.open('rb') as f:
        if digest(f)!=bundle['sha256']:raise ValueError('Archive SHA256 mismatch')
    if a.bundle.stat().st_size!=bundle['bytes']:raise ValueError('Archive size mismatch')
    expected={r['checkpoint']:r for r in json.loads(a.manifest.read_text())['weights']}
    root=a.output.resolve()
    with tarfile.open(a.bundle) as t:
        members=t.getmembers();names=[m.name for m in members]
        if len(names)!=len(set(names)) or len(names)!=bundle['checkpoints']:raise ValueError('Unexpected member count')
        # Validate every member before writing anything. Never use extractall.
        for m in members:
            path=PurePosixPath(m.name)
            if not m.isfile() or path.is_absolute() or '..' in path.parts or m.name not in expected:
                raise ValueError('Unsafe/unlisted archive member')
            dest=(root/m.name).resolve()
            if root not in dest.parents or dest.exists():raise ValueError('Unsafe/existing destination')
            with t.extractfile(m) as f:
                if digest(f)!=expected[m.name]['sha256'] or m.size!=expected[m.name]['bytes']:
                    raise ValueError('Checkpoint hash/size mismatch')
        for m in members:
            dest=root/m.name;dest.parent.mkdir(parents=True,exist_ok=True)
            with t.extractfile(m) as source,dest.open('xb') as out:
                for b in iter(lambda:source.read(1024*1024),b''):out.write(b)
    print('Verified and extracted',len(members),'checkpoints to',root)

if __name__=='__main__':main()
