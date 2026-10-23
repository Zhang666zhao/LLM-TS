import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import full_validation_server as full
import predict


class FullQueue(unittest.TestCase):
    def test_no_truth_access_during_requests_and_no_duplicate_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name in ('inputs','attempts','responses'):(root/name).mkdir()
            reused=['cached-'+str(i) for i in range(2815)]
            jobs=[]
            for target in (1,2):
                w=dict(window_id='w'+str(target),cutoff='t',timestamps=['s'],variables=['x'],history=[[1] for i in range(512)])
                (root/'inputs'/(str(target)+'.json')).write_text(json.dumps(w))
                jobs.append(dict(id=w['window_id'],target=target,input_hash=predict.digest(w)))
            (root/'budget.json').write_text(json.dumps(dict(deadline=time.time()+30,max_requests=2)))
            (root/'manifest.json').write_text(json.dumps(dict(jobs=jobs,reused=reused,variables=['x'])))
            cfg=json.loads((Path(full.__file__).parent/'configs/dao_pro_concurrent.json').read_text())
            (root/'config.json').write_text(json.dumps(cfg))
            def complete(config,messages):
                # There is deliberately no truth file available to inference.
                self.assertFalse((root/'truth.json').exists())
                return {'choices':[{'finish_reason':'stop','message':{'content':'{"labels":[{"variable":"x","L1":0,"L2":0}]}'}}]}
            with patch.dict(os.environ,{'TOKEN_SERVICE_API_KEY':'synthetic'}),patch.object(predict.ChatCompletions,'complete',side_effect=complete) as call,patch.object(full.subprocess,'run'):
                full.infer(root)
                self.assertEqual(call.call_count,2)
            status=json.loads((root/'status.json').read_text())
            self.assertEqual(status['valid'],2817)
            self.assertEqual(status['attempted'],2)
            self.assertEqual(len(list((root/'responses').glob('*.json'))),2)


if __name__=='__main__':unittest.main()
