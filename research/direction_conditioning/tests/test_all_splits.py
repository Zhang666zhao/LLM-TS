import sys
import unittest
import json
import os
import tempfile
import time
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from all_splits_server import SPLITS, anchor_for, retry_kind, causal_history
import all_splits_server as runner


class AllSplits(unittest.TestCase):
    def test_split_groups_never_cross_boundaries(self):
        expected = {'train':(8065,253),'val':(2817,89),'test':(2817,89)}
        total = 0
        for split,(first,last) in SPLITS.items():
            anchors = set()
            for t in range(first,last+1):
                a = anchor_for(split,t)
                self.assertTrue(first <= a <= t <= last)
                self.assertLess(t-a,32)
                self.assertLess(t+63,{'train':8640,'val':11520,'test':14400}[split])
                anchors.add(a)
            self.assertEqual((last-first+1,len(anchors)),expected[split])
            self.assertEqual(anchor_for(split,last),last)
            total += last-first+1
        self.assertEqual(total*7,95893)

    def test_outside_split_rejected(self):
        for split,(first,last) in SPLITS.items():
            for t in (first-1,last+1):
                with self.assertRaises(ValueError):anchor_for(split,t)

    def test_auth_and_contract_errors_are_not_retried(self):
        for message in ('API HTTP status 401; no automatic retry','API HTTP status 403; no automatic retry',
                        'API HTTP status 400; no automatic retry','deadline','fatal_ValueError'):
            self.assertEqual(retry_kind(message),'fatal')
        self.assertEqual(retry_kind('API HTTP status 429; no automatic retry'),'rate_limit')
        self.assertEqual(retry_kind('API HTTP status 503; no automatic retry'),'transient')
        self.assertEqual(retry_kind('anchor_joint_inconsistency'),'repair')

    def test_future_cannot_change_causal_inputs(self):
        try:
            import numpy as np
            import sklearn
        except ImportError:
            self.skipTest('Run numeric contract on existing server baseline environment')
        raw = np.random.default_rng(3).normal(size=(1000,7))
        history, scaler = causal_history(raw,700,800)
        changed = raw.copy();changed[700:] += 100000
        history2, scaler2 = causal_history(changed,700,800)
        self.assertTrue(np.array_equal(history,history2))
        self.assertTrue(np.array_equal(scaler.mean_,scaler2.mean_))
        later, later_scaler = causal_history(raw,900,800)
        self.assertTrue(np.allclose(later_scaler.mean_,raw[:800].mean(axis=0)))
        self.assertEqual(later.shape,(512,7))

    def queue_fixture(self, root, count, cap):
        for name in ('inputs','responses','attempts'):(root/name).mkdir()
        cfg = json.loads((Path(runner.__file__).parent/'configs/dao_pro_concurrent.json').read_text())
        jobs=[]
        for t in range(512,512+32*count,32):
            wid=f'ETTh1-train-{t}'
            w=dict(window_id=wid,cutoff='2000-01-01T00:00:00',timestamps=['2000-01-01T00:00:00'],
                   variables=['v'],history=[[0.] for _ in range(512)])
            (root/'inputs'/f'{wid}.json').write_text(json.dumps(w))
            jobs.append(dict(id=wid,split='train',target=t,input_hash=runner.predict.digest(w)))
        manifest=dict(anchors=jobs,reused=[],variables=['v'],config_sha256=runner.predict.digest(cfg),
                      prompt_sha256=runner.predict.digest(runner.PROMPT))
        budget=dict(deadline=time.time()+60,max_requests=cap,concurrency=30,send_interval_seconds=0,
                    max_attempts_per_anchor=3,rate_limit_cooldown_seconds=0)
        for name,obj in [('config',cfg),('manifest',manifest),('budget',budget)]:
            (root/(name+'.json')).write_text(json.dumps(obj))

    def test_request_cap_is_checked_inside_concurrent_admission(self):
        valid=dict(choices=[dict(finish_reason='stop',message=dict(content=json.dumps(
            dict(labels=[dict(variable='v',L1=0,L2=0)]))))])
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.queue_fixture(root,2,1)
            with patch.dict(os.environ,{'TOKEN_SERVICE_API_KEY':'test-only'}), \
                 patch.object(runner.predict.ChatCompletions,'complete',return_value=valid) as call, \
                 patch.object(runner.subprocess,'run'):
                runner.STOP=False;runner.infer(root)
            self.assertEqual(call.call_count,1)
            status=json.loads((root/'status.json').read_text())
            self.assertEqual(status['stop_reason'],'request_cap')
            self.assertEqual(status['attempted'],1)

    def test_invalid_repair_is_bounded_and_never_reads_truth(self):
        invalid=dict(choices=[dict(finish_reason='length',message=dict(content=''))])
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.queue_fixture(root,1,10)
            # No truth file exists: inference must still run and record every failure.
            with patch.dict(os.environ,{'TOKEN_SERVICE_API_KEY':'test-only'}), \
                 patch.object(runner.predict.ChatCompletions,'complete',return_value=invalid) as call, \
                 patch.object(runner.subprocess,'run'):
                runner.STOP=False;runner.infer(root)
            self.assertEqual(call.call_count,3)
            status=json.loads((root/'status.json').read_text())
            self.assertEqual(status['stop_reason'],'anchor_attempt_cap')
            self.assertEqual(len(list((root/'attempts').glob('*-response.json'))),3)


if __name__ == '__main__':unittest.main()
