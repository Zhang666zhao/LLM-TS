"""Verify standalone inference recipe against existing validation forecasts only."""
from common import *
from inference_recipe import Replay
import sys
name=sys.argv[1];indices=np.array([0,1,15,512,1024,2048,2815,2816]);z=np.load(RUN/'cache/trainval_z.npy');dates=json.loads((RUN/'cache/dates.json').read_text());orig=np.array(json.loads((RUN/'cache/window_manifest.json').read_text())['val'])[indices]
replay=Replay(z,dates,orig,RUN/'replay_checks'/name);p=replay.source('run:'+name);expected=np.load(ROOT/'runs'/name/'val_predictions.npy')[indices];delta=float(np.max(np.abs(p-expected)));result={'candidate':name,'validation_indices':indices.tolist(),'max_abs_delta':delta,'tolerance':1e-5,'code_revision':git_head(),'test_access':False};save_json(RUN/'replay_checks'/name/'check.json',result);print(result,flush=True);assert delta<1e-5,'Standalone replay differs'
