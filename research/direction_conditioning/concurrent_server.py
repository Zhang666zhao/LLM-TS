"""Bounded server-only concurrent inference; no retries and no truth access."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import os
from pathlib import Path
import time

import predict
from explore_server import pack, joint_valid
from finalize_server import PROMPT
from screen import save


def dispatch(jobs, worker, concurrency, max_requests, wall_seconds):
    """On first failure stop admission; already in-flight calls finish and count.

    Deadline controls admission. The worker must bound network time separately;
    a remote request cannot be reliably cancelled after it has been accepted.
    """
    if not 1 <= concurrency <= 50 or max_requests < 1 or wall_seconds <= 0:
        raise ValueError('Invalid concurrency/request/time bound')
    if len(jobs) > max_requests:
        raise ValueError('Manifest exceeds request budget; refusing partial silent dispatch')
    ids = [j['id'] for j in jobs]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate job IDs')
    deadline = time.monotonic() + wall_seconds
    completed = {}; admitted = []; pending = {}; stopped = False; index = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        while pending or (index < len(jobs) and not stopped):
            while not stopped and index < len(jobs) and len(pending) < concurrency:
                if time.monotonic() >= deadline:
                    stopped = True
                    break
                job = jobs[index]; index += 1
                pending[pool.submit(worker, job, deadline)] = job['id']
                admitted.append(job['id'])
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            # Process all available completions before admitting more work.
            for future in done:
                job_id = pending.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    result = dict(status='failed', error_type=type(error).__name__)
                completed[job_id] = result
                if result.get('status') != 'valid':
                    stopped = True
    return dict(results=completed, admitted=admitted,
                unsent=[j['id'] for j in jobs if j['id'] not in admitted],
                status='complete' if len(completed)==len(jobs) and all(r['status']=='valid' for r in completed.values()) else 'incomplete')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True,help='Directory with history-only window JSONs')
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--concurrency',type=int,default=50)
    parser.add_argument('--max-requests',type=int,required=True)
    parser.add_argument('--wall-seconds',type=int,required=True)
    parser.add_argument('--send',action='store_true')
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    if set(config)-{'adapter','base_url','api_key_env','model','timeout_seconds','max_tokens','temperature','label','include_stats','include_semantics'}:
        raise ValueError('Unexpected config fields; no credentials in config')
    if config['label'] != json.loads((Path(__file__).parent/'configs/dao.json').read_text())['label']:
        raise ValueError('Compact pack supports frozen tau0.5 L1/L2 only')
    if args.send:
        expected=Path('/nfs/dataset-ofs-661-142656/qiminhao/Foundation_model/agentic_TS/TSRAG')
        if not expected.is_dir() or not Path('/etc/ssl/certs/ca-certificates.crt').is_file():
            raise ValueError('Sending allowed only in configured server workspace')
        if not os.environ.get(config['api_key_env']):raise ValueError('Missing credential environment variable')
        os.environ['SSL_CERT_FILE']='/etc/ssl/certs/ca-certificates.crt'
    jobs=[]; seen=set()
    for path in sorted(args.inputs.glob('*.json')):
        window=json.loads(path.read_text())
        predict.evidence(window,config)
        if len(window['history'])!=512:raise ValueError('Expected512 history steps')
        digest=predict.digest(window)
        if digest in seen:raise ValueError('Duplicate input content')
        seen.add(digest)
        payload=pack(window)
        jobs.append(dict(id=window['window_id'],input_hash=digest,payload=payload))
    if not jobs or len(jobs)>args.max_requests or len({j['id'] for j in jobs})!=len(jobs):
        raise ValueError('Empty, duplicate or over-budget window manifest')
    if not 1<=args.concurrency<=50 or args.wall_seconds<=0:raise ValueError('Invalid limits')
    args.output.mkdir(parents=True,mode=0o700,exist_ok=False)
    save(args.output/'manifest.json',dict(config=config,concurrency=args.concurrency,
         max_requests=args.max_requests,wall_seconds=args.wall_seconds,retries=0,
         jobs=[{k:j[k] for k in ('id','input_hash')} for j in jobs],prompt=PROMPT,
         warning='Deadline stops admission; accepted remote calls may finish afterwards. No guaranteed throughput or token pricing.'))
    index={j['id']:i for i,j in enumerate(jobs)}
    if not args.send:
        save(args.output/'status.json',dict(status='dry_run',requests_sent=0));return
    def worker(job,deadline):
        number=index[job['id']]
        remaining=deadline-time.monotonic()
        if remaining<=0:return dict(status='not_sent_deadline')
        messages=[dict(role='system',content=PROMPT),dict(role='user',content=json.dumps(job['payload'],separators=(',',':')))]
        save(args.output/f'attempt-{number}.json',dict(id=job['id'],input_hash=job['input_hash'],messages=messages,at=time.time()))
        started=time.monotonic()
        try:
            raw=predict.ChatCompletions().complete(dict(config,timeout_seconds=min(config['timeout_seconds'],remaining)),messages)
            result=dict(status='invalid',usage=raw.get('usage'),raw_response=raw)
            choice=raw['choices'][0]
            if choice['finish_reason']=='stop':
                prediction=predict.validate_response(choice['message']['content'],list(job['payload']['variables']))
                if joint_valid(prediction,job['payload']):result.update(status='valid',prediction=prediction)
        except Exception as error:
            result=locals().get('result',{})
            result.update(status='failed',error_type=type(error).__name__)
        result.update(id=job['id'],seconds=time.monotonic()-started)
        save(args.output/f'response-{number}.json',result)
        return {k:v for k,v in result.items() if k!='raw_response'}
    summary=dispatch(jobs,worker,args.concurrency,args.max_requests,args.wall_seconds)
    save(args.output/'status.json',summary)
    print(json.dumps(dict(status=summary['status'],admitted=len(summary['admitted']),unsent=len(summary['unsent']))))


if __name__=='__main__':main()
