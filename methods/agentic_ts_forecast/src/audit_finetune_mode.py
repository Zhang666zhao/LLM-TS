"""Read-only checkpoint diagnostic of train/eval stochasticity; never fit or score test."""
from common import *
import torch
from chronos import Chronos2Pipeline
out=RUN/'audits/finetune_mode';out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);z=np.load(RUN/'cache/trainval_z.npy');inputs=[torch.from_numpy(z[8640-512:8640].T.copy())];parent=ROOT/'runs/20261105_chronosft_lr5e7_mse/trainer/finetuned-ckpt';pipe=Chronos2Pipeline.from_pretrained(str(parent),device_map='cuda:0');initial=pipe.model.training

def predict():return pipe.predict_quantiles(inputs,prediction_length=64,batch_size=112,context_length=512,cross_learning=False)[0][0].cpu().numpy()
pipe.model.eval();a=predict();b=predict();pipe.model.train();c=predict();d=predict();pipe.model.eval();e=predict();m={'checkpoint':str(parent),'default_reloaded_training_mode':initial,'eval_repeat_max_delta':float(np.max(abs(a-b))),'train_repeat_max_delta':float(np.max(abs(c-d))),'eval_train_max_delta':float(np.max(abs(a-c))),'eval_restored_max_delta':float(np.max(abs(a-e))),'dropout_rate':pipe.model.config.dropout_rate,'test_access':False,'training_performed':False,'code_revision':git_head()};save_json(out/'diagnostic.json',m);print(m,flush=True);assert m['eval_repeat_max_delta']<1e-6 and m['eval_restored_max_delta']<1e-6
