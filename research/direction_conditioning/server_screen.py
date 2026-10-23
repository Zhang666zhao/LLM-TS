"""Server-only bounded screening. Secret arrives on SSH stdin, never on disk."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import predict
from screen import save, score, MODELS


def main():
    root = Path(__file__).resolve().parent
    run = Path(sys.argv[1]).resolve()
    if run.exists():
        raise ValueError('Never reuse a run directory')
    secret = json.load(sys.stdin)['key']
    os.environ['TOKEN_SERVICE_API_KEY'] = secret
    del secret
    os.environ['SSL_CERT_FILE'] = '/etc/ssl/certs/ca-certificates.crt'
    run.mkdir(parents=True, mode=0o700)
    start = time.monotonic()
    save(run/'budget.json', dict(start=time.time(), wall_seconds=900, max_requests=4,
                                max_output_tokens=8192, retries=0))
    config = json.loads((root/'configs/dao.json').read_text())
    config.update(max_tokens=8192, timeout_seconds=180)
    package = json.loads(subprocess.check_output([sys.executable, str(root/'scripts/export_etth1.py')], text=True, timeout=60))
    save(run/'data_version.json', package['provenance'])
    save(run/'truth.json', package['truth'])
    for window in package['windows']:
        predict.evidence(window, config)
        save(run/(window['window_id']+'.json'), window)
    manifest = dict(models=MODELS, windows=[w['window_id'] for w in package['windows']],
                    window_hashes={w['window_id']:predict.digest(w) for w in package['windows']},
                    config=config, host=socket.gethostname(), python=sys.version, executable=sys.executable,
                    code_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in root.rglob('*.py')}, screening_only=True,
                    baseline_commit='dbea4ff', code_state='uncommitted snapshot identified by hashes',
                    remote_dataset_code_sha256=package['source_code_sha256'], cost=None,
                    selection='validation first and last windows, no label-based selection')
    save(run/'manifest.json', manifest)
    # Release offline package (including truth); request loop only opens historical files.
    del package
    attempted = 0
    try:
        for index, model in enumerate(MODELS):
            for wid in manifest['windows']:
                if attempted >= 4 or time.monotonic()-start > 720:
                    raise RuntimeError('Insufficient time in 900s budget')
                window = json.loads((run/(wid+'.json')).read_text())
                assert predict.digest(window) == manifest['window_hashes'][wid]
                attempted += 1
                save(run/('attempt-'+str(attempted)+'.json'), dict(model=model, window=wid, at=time.time()))
                result = predict.run(window, dict(config, model=model), send=True)
                save(run/(str(index)+'-'+wid+'-response.json'), result)
                print(json.dumps(dict(attempt=attempted, status=result['status'], model=model, window=wid)), flush=True)
                if result['status'] != 'valid':
                    raise RuntimeError('Invalid response; fail stop')
        save(run/'status.json', dict(status='complete', attempted=attempted, elapsed=time.monotonic()-start))
    except Exception as error:
        save(run/'status.json', dict(status='failed', attempted=attempted, error_type=type(error).__name__, elapsed=time.monotonic()-start))
        print('Stopped without retries', flush=True)
    finally:
        os.environ.pop('TOKEN_SERVICE_API_KEY', None)
    score(run)


if __name__ == '__main__':
    main()
