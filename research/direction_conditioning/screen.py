"""Bounded four-call screening. Separate prepare, send (history only), score.
No prompt search, no retry, no training. Run directory is immutable once prepared.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import predict

ROOT = Path(__file__).resolve().parent
MODELS = ['tencent/DeepSeek-V4-Flash', 'aliyun/DeepSeek-v4-pro']


def save(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)


def metric(truth, predicted):
    # None is an invalid prediction and counts as FN/incorrect, never as zero.
    if len(truth) != len(predicted) or not truth:
        raise ValueError('Mismatched/empty scoring inputs')
    rows = {}
    for cls in (-1, 0, 1):
        tp = sum(a == cls and b == cls for a, b in zip(truth, predicted))
        fp = sum(a != cls and b == cls for a, b in zip(truth, predicted))
        fn = sum(a == cls and b != cls for a, b in zip(truth, predicted))
        rows[str(cls)] = dict(support=truth.count(cls), tp=tp, fp=fp, fn=fn,
                             f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return dict(n=len(truth), macro_f1=sum(r['f1'] for r in rows.values())/3,
                accuracy=sum(a == b for a, b in zip(truth, predicted))/len(truth),
                invalid=sum(b is None for b in predicted), per_class=rows,
                confusion={str(a): {str(b): sum(x == a and y == b for x, y in zip(truth, predicted))
                                      for b in (-1, 0, 1, None)} for a in (-1, 0, 1)})


def prepare(run):
    if run.exists():
        raise ValueError('Refuse to overwrite prepared run')
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes', '-o', 'ConnectTimeout=10',
               'GPU_Foundation', '/nfs/dataset-ofs-661-142656/qiminhao/Env/tsrag_baselines_20260926/bin/python -']
    export = ROOT/'scripts/export_etth1.py'
    result = subprocess.run(command, input=export.read_text(), text=True, capture_output=True, timeout=60, check=True)
    package = json.loads(result.stdout)
    run.mkdir(parents=True, mode=0o700)
    config = json.loads((ROOT/'configs/dao.json').read_text())
    for window in package['windows']:
        predict.evidence(window, config)
        save(run/(window['window_id']+'.json'), window)
    save(run/'truth.json', package['truth'])
    save(run/'data_version.json', package['provenance'])
    code = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'predict.py', ROOT/'screen.py', export, ROOT/'configs/dao.json')}
    save(run/'manifest.json', dict(models=MODELS, windows=[w['window_id'] for w in package['windows']],
                                  window_hashes={w['window_id']: predict.digest(w) for w in package['windows']},
                                  max_requests=4, wall_seconds=900, max_output_tokens_per_call=config['max_tokens'],
                                  expected_calls=4, config=config, code_hashes=code, python=platform.python_version(),
                                  git_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                                  git_dirty_allowed=True, screening_only=True, cost_limit=None,
                                  remote_code_sha256=package['source_code_sha256'],
                                  sampling='first and last validation target windows chosen without labels',
                                  launch_command='python3 research/direction_conditioning/screen.py send --run '+str(run)))


def send(run):
    manifest = json.loads((run/'manifest.json').read_text())
    config = manifest['config']
    if not os.environ.get(config['api_key_env']):
        raise ValueError('Missing API key environment variable; zero requests sent')
    for name, expected in manifest['code_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Code changed after prepare; do not silently change this run')
    # This phase does not open truth.json or data_version.json.
    save(run/'started.json', dict(started_at=time.time(), deadline=time.time()+900, max_requests=4))
    start = time.monotonic()
    attempted = 0
    try:
        for model_index, model in enumerate(manifest['models']):
            for window_id in manifest['windows']:
                if attempted >= 4 or time.monotonic()-start >= 900:
                    raise RuntimeError('Budget reached')
                window = json.loads((run/(window_id+'.json')).read_text())
                if predict.digest(window) != manifest['window_hashes'][window_id]:
                    raise ValueError('Window hash mismatch')
                config = dict(config, model=model)
                attempted += 1
                save(run/('attempt-'+str(attempted)+'.json'), dict(model=model, window=window_id, at=time.time()))
                record = predict.run(window, config, send=True)
                save(run/(str(model_index)+'-'+window_id+'-response.json'), record)
                if record['status'] != 'valid':
                    raise RuntimeError('Invalid response; fail stop, no retry')
        save(run/'status.json', dict(status='complete', attempted=attempted, elapsed=time.monotonic()-start))
    except Exception as error:
        save(run/'status.json', dict(status='failed', error_type=type(error).__name__, attempted=attempted,
                                     elapsed=time.monotonic()-start))
        raise RuntimeError('Screen failed; inspect private records, no automatic retry') from None


def score(run):
    manifest = json.loads((run/'manifest.json').read_text())
    truth = json.loads((run/'truth.json').read_text())
    results = {}
    for index, model in enumerate(manifest['models']):
        results[model] = {}
        for label in ('L1', 'L2'):
            actual, predictions = [], []
            for window in manifest['windows']:
                path = run/(str(index)+'-'+window+'-response.json')
                record = json.loads(path.read_text()) if path.exists() else {}
                rows = {r['variable']: r for r in record.get('prediction', {}).get('labels', [])} if record.get('status') == 'valid' else {}
                for row in truth[window]:
                    actual.append(row[label])
                    predictions.append(rows.get(row['variable'], {}).get(label))
            results[model][label] = metric(actual, predictions)
    actual = {k: [r[k] for w in manifest['windows'] for r in truth[w]] for k in ('L1', 'L2')}
    save(run/'metrics.json', dict(models=results, always_zero={k: metric(v, [0]*len(v)) for k, v in actual.items()},
                                  warning='Only 2 windows x 7 correlated variables; not evidence of generalization. Missing classes F1=0.'))
    print(json.dumps(results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'send', 'score'])
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'send': send, 'score': score}[args.phase](args.run.resolve())
