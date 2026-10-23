"""Frozen ETTh1 train/val/test labels; causal stride32, cached val, bounded API work.

prepare reads targets only for separate truth/audit files; infer reads history only.
Scoring runs in a separate process without the API credential after dispatch ends.
"""
import argparse
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

import predict
from explore_server import pack, joint_valid
from finalize_server import PROMPT
from full_validation_server import PROJECT, atomic, messages
from screen import save, metric

ROOT = Path(__file__).resolve().parent
PRIOR = PROJECT / 'runs/20261019_dao_etth1_stride32_v5'
SPLITS = {'train': (512, 8576), 'val': (8640, 11456), 'test': (11520, 14336)}
DATA_SHA = '3dbe45a7a25cbf0eb6d82cb84ef4487ea2b73a9341e5ff0d082e340b9d943f1d'
STOP = False


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def anchor_for(split, target):
    first, last = SPLITS[split]
    if not first <= target <= last:
        raise ValueError('Target outside frozen split')
    return first + 32 * ((target - first) // 32)


def causal_history(raw, target, train_end=8640):
    """A train prefix is fitted only through target-1; later splits freeze it.

    No value after fit_end is read by this function. This generalizes the original
    validation scaler without leaking later training observations to early inputs.
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(raw[:min(target, train_end)])
    return scaler.transform(raw[target-512:target]).astype('float32'), scaler


def retry_kind(error):
    if error in ('anchor_joint_inconsistency', 'incomplete_response', 'invalid_schema', 'timeout'):
        return 'repair'
    if error.startswith('API HTTP status 429;'):
        return 'rate_limit'
    if any(error.startswith('API HTTP status ' + str(s) + ';') for s in (500, 502, 503, 504)):
        return 'transient'
    if error.startswith('API connection failed;'):
        return 'transient'
    return 'fatal'


def prepare(run):
    if run.exists():
        raise ValueError('Run already exists; no overwrite or budget reset')
    start = time.time()
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(PROJECT / 'research/multivariate_oracle_six'))
    from dataset import load_data, labels
    values, _, info = load_data(PROJECT, 'ETTh1')
    assert info['source_sha256'] == DATA_SHA
    frame = pd.read_csv(PROJECT / 'artifacts/baseline_raw_v1/ETTh1.csv')
    raw = frame[info['variables']].to_numpy(dtype=np.float64)
    dates = pd.to_datetime(frame['date'])
    assert dates.is_unique and dates.is_monotonic_increasing
    assert (dates.diff().dropna() == pd.Timedelta(hours=1)).all()
    cfg = json.loads((ROOT / 'configs/dao_pro_concurrent.json').read_text())
    old = json.loads((PRIOR / 'manifest.json').read_text())
    assert cfg == json.loads((PRIOR / 'config.json').read_text()) and old['prompt'] == PROMPT
    assert json.loads((PRIOR / 'metrics.json').read_text())['complete']
    run.mkdir(parents=True, mode=0o700)
    for name in ('inputs', 'responses', 'attempts', 'code', 'labels'):
        (run / name).mkdir(mode=0o700)
    (run / 'code/configs').mkdir(mode=0o700)
    for f in ROOT.glob('*.py'):
        shutil.copy2(f, run / 'code' / f.name)
    for f in (ROOT / 'configs').glob('*.json'):
        shutil.copy2(f, run / 'code/configs' / f.name)
    truth, window_map, anchors, reused, normalization_audit = {}, [], [], [], []
    for split, (first, last) in SPLITS.items():
        assert first == info['split'][split + '_start'] + 512
        assert last == info['split'][split + '_end'] - 64
        for target in range(first, last + 1):
            wid = f'ETTh1-{split}-{target}'
            x, y = values[target-512:target], values[target:target+64]
            assert (x.astype(np.float64).std(axis=0) > .001).all(), 'Original epsilon floor active'
            l1, l2 = labels(x, y)
            truth[wid] = [dict(variable=n, L1=int(a), L2=int(b))
                          for n, a, b in zip(info['variables'], l1, l2)]
            window_map.append(dict(id=wid, split=split, target=target,
                                   anchor=anchor_for(split, target), age=target-anchor_for(split, target)))
        for target in range(first, last + 1, 32):
            wid = f'ETTh1-{split}-{target}'
            x, scaler = causal_history(raw, target)
            assert (x.astype(np.float64).std(axis=0) > .001).all(), 'Causal epsilon floor active'
            # Offline equivalence audit only; transformed future is never serialized into inputs.
            causal_y = scaler.transform(raw[target:target+64]).astype(np.float32)
            causal_z = labels(x, causal_y)
            fixed_z = labels(values[target-512:target], values[target:target+64])
            assert all(np.array_equal(a, b) for a, b in zip(causal_z, fixed_z)), 'Label boundary mismatch'
            if split != 'train':
                assert np.array_equal(x, values[target-512:target]), 'Frozen scaler mismatch'
            w = dict(window_id=wid, cutoff=dates.iloc[target-1].isoformat(),
                     timestamps=[t.isoformat() for t in dates.iloc[target-512:target]],
                     variables=info['variables'], history=x.tolist())
            predict.evidence(w, cfg)
            digest = predict.digest(w)
            save(run / 'inputs' / f'{wid}.json', w)
            anchors.append(dict(id=wid, split=split, target=target, input_hash=digest))
            normalization_audit.append(dict(id=wid, fit_start=0, fit_end_exclusive=min(target, 8640),
                                            mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                                            anchor_true_labels_unchanged=True, epsilon_floor_inactive=True))
            if split == 'val':
                source = PRIOR / 'responses' / f'{target}.json'
                r = json.loads(source.read_text())
                assert r['status'] == 'valid' and r['input_hash'] == digest
                assert json.loads((PRIOR / 'inputs' / f'{target}.json').read_text()) == w
                pred = predict.validate_response(json.dumps(r['prediction']), info['variables'])
                assert joint_valid(pred, pack(w))
                save(run / 'responses' / f'{wid}.json', dict(id=wid, split=split, target=target,
                     status='valid', prediction=pred, reused=True, input_hash=digest,
                     source=str(source), source_sha256=sha(source)))
                reused.append(wid)
    assert len(window_map) == 13699 and len(anchors) == 431 and len(reused) == 89
    save(run / 'truth.json', truth)
    save(run / 'data_version.json', info)
    save(run / 'config.json', cfg)
    save(run / 'normalization_audit.json', normalization_audit)
    save(run / 'budget.json', dict(start=start, deadline=start+7200, wall_seconds=7200,
         unique_new_anchors=342, max_requests=410, max_attempts_per_anchor=3,
         concurrency=30, send_interval_seconds=10, rate_limit_cooldown_seconds=60,
         max_tokens_per_request=16384, gpu_count=0, cost=None,
         retry_policy='Only transient/429/schema/incomplete/joint-conflict, identical request; no success repeated'))
    save(run / 'manifest.json', dict(dataset='ETTh1', splits=SPLITS, context=512, horizon=64,
         window_stride=1, label_reuse_stride=32, variables=info['variables'], anchors=anchors,
         window_map=window_map, reused=reused, total_windows=13699, total_rows=95893,
         prompt=PROMPT, prompt_sha256=predict.digest(PROMPT), config_sha256=predict.digest(cfg),
         source_validation=str(PRIOR), source_validation_csv_sha256=sha(PRIOR/'window_variable_results.csv'),
         input_normalization='StandardScaler fit raw[0:min(anchor_target,8640)], history-only prefix; frozen after train',
         truth_normalization='Unchanged original training-fitted scaler; epsilon inactive, anchor labels checked equal',
         source_data_code_sha256=sha(PROJECT/'research/multivariate_oracle_six/dataset.py'),
         code_sha256={str(p.relative_to(run/'code')):sha(p) for p in (run/'code').rglob('*') if p.is_file()},
         host=socket.gethostname(), python=sys.version,
         selection='Frozen prior Pro/concise prompt; no score-based selection or test tuning',
         warning='Val used in development. Shared labels/windows are correlated. No downstream training in this run.'))
    atomic(run / 'status.json', dict(status='prepared', valid=89, total_anchors=431,
                                    by_split={'train':0, 'val':89, 'test':0}, attempted=0, pending=342))
    print(json.dumps(dict(prepared=True, anchors=431, reused=89, new_anchors=342,
                          rows=95893, deadline=start+7200)), flush=True)


def launch(run):
    b = json.loads((run/'budget.json').read_text())
    m = json.loads((run/'manifest.json').read_text())
    assert time.time() < b['deadline'] and json.loads((run/'status.json').read_text())['status'] == 'prepared'
    for name, expected in m['code_sha256'].items():
        assert sha(run/'code'/name) == expected, 'Snapshot changed'
    key = json.load(sys.stdin)['key']
    if not isinstance(key, str) or not key.strip():
        raise ValueError('Missing credential; no launch')
    env = dict(os.environ, TOKEN_SERVICE_API_KEY=key, SSL_CERT_FILE='/etc/ssl/certs/ca-certificates.crt')
    del key
    command = [sys.executable, str(run/'code/all_splits_server.py'), 'run', '--run', str(run)]
    save(run/'launch.json', dict(at=time.time(), command=command, host=socket.gethostname()))
    fd = os.open(run/'launcher.log', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as log:
        child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    env.pop('TOKEN_SERVICE_API_KEY', None)
    save(run/'pid.json', dict(pid=child.pid))
    print(json.dumps(dict(launched=True, pid=child.pid, deadline=b['deadline'])), flush=True)


def infer(run):
    global STOP
    def stop(*_):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    save(run/'run_lock.json', dict(pid=os.getpid(), at=time.time()))
    m = json.loads((run/'manifest.json').read_text())
    b = json.loads((run/'budget.json').read_text())
    cfg = json.loads((run/'config.json').read_text())
    assert predict.digest(cfg) == m['config_sha256'] and predict.digest(PROMPT) == m['prompt_sha256']
    if not os.environ.get(cfg['api_key_env']):
        raise ValueError('Missing credential')
    todo = deque(j for j in m['anchors'] if j['id'] not in set(m['reused']))
    attempts = Counter()
    counts = Counter({'val':89, 'train':0, 'test':0})
    attempted = finished = failures = 0
    pending = {}
    stop_reason = None
    next_send = 0.
    interval = b['send_interval_seconds']
    terminal_ids = []

    def progress(state):
        atomic(run/'status.json', dict(status=state, valid=sum(counts.values()), by_split=dict(counts),
             total_anchors=431, total_windows=13699, attempted=attempted, finished=finished,
             failed_attempts=failures, terminal_ids=terminal_ids, in_flight=len(pending), pending=len(todo),
             send_interval_seconds=interval, next_send_at=next_send, stop_reason=stop_reason,
             updated_at=time.time(), deadline=b['deadline']))

    def worker(j, number):
        wid = j['id']
        w = json.loads((run/'inputs'/f'{wid}.json').read_text())
        if predict.digest(w) != j['input_hash']:
            raise ValueError('Input hash mismatch')
        payload = pack(w)
        msgs = messages(payload)
        ap = run/'attempts'/f'{wid}-{number}.json'
        request_hash = predict.digest(dict(config=cfg, messages=msgs))
        save(ap, dict(id=wid, attempt=number, at=time.time(), request_hash=request_hash,
                      input_hash=j['input_hash'], status='sent'))
        start = time.time()
        r = dict(id=wid, split=j['split'], target=j['target'], attempt=number,
                 input_hash=j['input_hash'], request_hash=request_hash, status='invalid', reused=False)
        error = None
        try:
            remaining = b['deadline'] - time.time()
            if remaining <= 0:
                raise RuntimeError('deadline')
            raw = predict.ChatCompletions().complete(dict(cfg, timeout_seconds=min(remaining,cfg['timeout_seconds'])),msgs)
            r.update(raw_response=raw, usage=raw.get('usage'))
            c = raw['choices'][0]
            if c.get('finish_reason') != 'stop':
                error = 'incomplete_response'
            else:
                pred = predict.validate_response(c['message']['content'], m['variables'])
                if not joint_valid(pred, payload):
                    error = 'anchor_joint_inconsistency'
                else:
                    r.update(status='valid', prediction=pred)
        except RuntimeError as e:
            error = str(e)
        except (TimeoutError, socket.timeout):
            error = 'timeout'
        except (ValueError, KeyError, IndexError, TypeError):
            error = 'invalid_schema'
        except Exception as e:
            error = 'fatal_' + type(e).__name__
        r.update(seconds=time.time()-start, error=error)
        save(run/'attempts'/f'{wid}-{number}-response.json', r)
        atomic(run/'responses'/f'{wid}.json', r)
        return r['status'], error

    try:
        with ThreadPoolExecutor(max_workers=b['concurrency']) as pool:
            while pending or (todo and stop_reason is None):
                if STOP:
                    stop_reason = 'user_stop'
                if time.time() >= b['deadline']:
                    stop_reason = 'deadline'
                if todo and attempted >= b['max_requests']:
                    stop_reason = 'request_cap'
                while (todo and len(pending) < b['concurrency'] and stop_reason is None
                       and attempted < b['max_requests'] and time.time() >= next_send):
                    j = todo.popleft()
                    attempts[j['id']] += 1
                    attempted += 1
                    pending[pool.submit(worker,j,attempts[j['id']])] = j
                    next_send = time.time()+interval
                progress('running' if stop_reason is None else 'draining')
                if not pending:
                    if todo and stop_reason is None:
                        time.sleep(min(1, max(.05,next_send-time.time())))
                        continue
                    break
                done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for f in done:
                    j = pending.pop(f)
                    finished += 1
                    state, error = f.result()
                    if state == 'valid':
                        counts[j['split']] += 1
                    else:
                        failures += 1
                        kind = retry_kind(error)
                        if kind == 'fatal':
                            stop_reason = error
                            terminal_ids.append(j['id'])
                        elif attempts[j['id']] >= b['max_attempts_per_anchor']:
                            stop_reason = 'anchor_attempt_cap'
                            terminal_ids.append(j['id'])
                        elif stop_reason is None:
                            todo.append(j)
                            if kind == 'rate_limit':
                                interval = min(30, interval*1.5)
                                next_send = max(next_send,time.time()+b['rate_limit_cooldown_seconds'])
                            elif kind == 'transient':
                                next_send = max(next_send,time.time()+30)
            progress('complete' if sum(counts.values()) == 431 else 'stopped')
    except Exception as e:
        stop_reason = 'fatal_' + type(e).__name__
        progress('failed')
        raise
    finally:
        os.environ.pop('TOKEN_SERVICE_API_KEY', None)
    subprocess.run([sys.executable,str(Path(__file__).resolve()),'score','--run',str(run)],check=True)


def score(run):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    m = json.loads((run/'manifest.json').read_text())
    truth = json.loads((run/'truth.json').read_text())
    status = json.loads((run/'status.json').read_text())
    records = {p.stem:json.loads(p.read_text()) for p in (run/'responses').glob('*.json')}
    fields = ['split','window_id','target_start','anchor_window_id','anchor_target_start','label_age_steps',
              'variable','status','cached_anchor','L1_true','L1_pred','L1_correct','L2_true','L2_pred','L2_correct']
    metrics = {}
    exports = {}
    for split in SPLITS:
        windows = [j for j in m['window_map'] if j['split'] == split]
        a = {lab:dict(y=[],p=[],ay=[],ap=[],per={n:([],[]) for n in m['variables']}) for lab in ('L1','L2')}
        labels = {'l1':[], 'l2':[]}
        with (run/f'{split}_window_variable_results.csv').open('x',newline='') as cf, (run/f'{split}_window_variable_results.jsonl').open('x') as jf:
            os.chmod(cf.name,0o600); os.chmod(jf.name,0o600)
            writer = csv.DictWriter(cf,fieldnames=fields);writer.writeheader()
            for j in windows:
                aid = f"ETTh1-{split}-{j['anchor']}"
                r = records.get(aid,{})
                state = r.get('status','not_sent')
                pred = {v['variable']:v for v in r.get('prediction',{}).get('labels',[])} if state == 'valid' else {}
                for lab in ('L1','L2'):
                    labels[lab.lower()].append([pred.get(n,{}).get(lab,127) for n in m['variables']])
                for t in truth[j['id']]:
                    n = t['variable']
                    row = dict(split=split,window_id=j['id'],target_start=j['target'],anchor_window_id=aid,
                               anchor_target_start=j['anchor'],label_age_steps=j['age'],variable=n,
                               status=state,cached_anchor=r.get('reused',False))
                    for lab in ('L1','L2'):
                        y, p = t[lab], pred.get(n,{}).get(lab)
                        a[lab]['y'].append(y); a[lab]['p'].append(p)
                        a[lab]['per'][n][0].append(y);a[lab]['per'][n][1].append(p)
                        if j['age'] == 0:
                            a[lab]['ay'].append(y);a[lab]['ap'].append(p)
                        row.update({lab+'_true':y,lab+'_pred':p,lab+'_correct':y==p})
                    writer.writerow(row);jf.write(json.dumps(row)+'\n')
        metrics[split] = {}
        for lab, z in a.items():
            r = metric(z['y'],z['p'])
            r.update(per_variable={n:metric(*v) for n,v in z['per'].items()},
                     anchor_only=metric(z['ay'],z['ap']),always_zero=metric(z['y'],[0]*len(z['y'])))
            if r['invalid'] == 0:
                assert abs(f1_score(z['y'],z['p'],labels=[-1,0,1],average='macro',zero_division=0)-r['macro_f1'])<1e-12
                assert abs(accuracy_score(z['y'],z['p'])-r['accuracy'])<1e-12
            metrics[split][lab] = r
        complete = all(metrics[split][lab]['invalid'] == 0 for lab in ('L1','L2'))
        if complete:
            target = np.array([j['target'] for j in windows],dtype=np.int64)
            path = run/'labels'/f'{split}.npz'
            with path.open('xb') as f:
                os.chmod(path,0o600)
                np.savez_compressed(f,starts=target-512,target_starts=target,
                                    l1=np.array(labels['l1'],dtype=np.int8),l2=np.array(labels['l2'],dtype=np.int8),
                                    anchor_target_starts=np.array([j['anchor'] for j in windows],dtype=np.int64),
                                    label_age_steps=np.array([j['age'] for j in windows],dtype=np.int8))
            exports[split] = dict(path=str(path),sha256=sha(path),windows=len(windows))
    # Val must remain exactly the already-reported predictions and metrics.
    prior = json.loads((PRIOR/'metrics.json').read_text())
    for lab in ('L1','L2'):
        for key in ('macro_f1','accuracy'):
            assert metrics['val'][lab][key] == prior['metrics'][lab][key]
    usage = Counter()
    for p in (run/'attempts').glob('*-response.json'):
        for k,v in (json.loads(p.read_text()).get('usage') or {}).items():
            if type(v) is int:
                usage[k] += v
    complete = len(exports) == 3 and status['valid'] == 431
    save(run/'labels/manifest.json',dict(complete=complete,splits=exports,variables=m['variables'],
         source_data_sha256=DATA_SHA,label_source='LLM prediction; true labels excluded from NPZ',
         label_version='direction-l1-l2-tau05-stride32-causal-prefix-v1',source_run=str(run),
         history=512,horizon=64,label_reuse_stride=32,input_normalization=m['input_normalization']))
    result = dict(complete=complete,valid_anchors=status['valid'],total_anchors=431,total_windows=13699,
                  rows=95893,metrics=metrics,new_request_usage=dict(usage),attempted=status['attempted'],
                  failed_attempts=status['failed_attempts'],exports=exports,
                  elapsed_seconds=time.time()-json.loads((run/'budget.json').read_text())['start'])
    save(run/'metrics.json',result)
    atomic(run/'completion.json',dict(at=time.time(),complete=complete,metrics_ready=True))
    print(json.dumps(dict(complete=complete,metrics={s:{k:{f:v[f] for f in ('macro_f1','accuracy','invalid')}
          for k,v in labs.items()} for s,labs in metrics.items()})),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['prepare','launch','run','score'])
    p.add_argument('--run',type=Path,required=True)
    args = p.parse_args()
    if not PROJECT.is_dir():
        raise RuntimeError('Server project required; never send API requests from local workstation')
    {'prepare':prepare,'launch':launch,'run':infer,'score':score}[args.phase](args.run.resolve())
