"""Download pinned public ETTh1, or verify an authorized frozen local copy.

No training, API requests, preprocessing, or overwriting existing data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

PUBLIC_SHA = 'f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066'
FROZEN_SHA = '3dbe45a7a25cbf0eb6d82cb84ef4487ea2b73a9341e5ff0d082e340b9d943f1d'
URL = 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small/ETTh1.csv'

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('data/raw/ETTh1.csv'))
    p.add_argument('--verify-frozen', type=Path, help='Only verify this existing authorized historical file')
    args = p.parse_args()
    if args.verify_frozen:
        actual = hashlib.sha256(args.verify_frozen.read_bytes()).hexdigest()
        if actual != FROZEN_SHA:
            raise SystemExit('Frozen data SHA256 mismatch; do not replay historical experiments.')
        print('Historical frozen data SHA256 verified.')
        return
    if args.output.exists() or args.output.with_suffix('.source.json').exists():
        raise SystemExit('Output or provenance exists; refusing overwrite.')
    with urllib.request.urlopen(URL, timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != PUBLIC_SHA:
        raise SystemExit('Public source SHA256 mismatch; no data written.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as f:
        f.write(data)
    with args.output.with_suffix('.source.json').open('x') as f:
        json.dump(dict(url=URL, sha256=PUBLIC_SHA, historical_sha256=FROZEN_SHA,
                       historical_byte_identity=False), f, indent=2)
    print(str(args.output))
    print('Public file verified. NOT byte-identical to the historical frozen file; see data/RESEARCH_DATA.md.')

if __name__ == '__main__':
    main()
