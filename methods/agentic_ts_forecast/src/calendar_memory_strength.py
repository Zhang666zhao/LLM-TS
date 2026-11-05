"""Three residual strengths on only MULL/LUFL; exact calendar prior control."""
from common import *
from blend_sources import get_source
from view_prior import predict
from masked_residual_strength import apply
import sys
prior='20261105_calendarprior_exact_mse';pp=ROOT/'runs'/prior;tools=['run:20261105_transformedcalft_'+v+'_mse' for v in ('semantic','pca','random')]
assert not pp.exists();pp.mkdir();subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',prior],check=True);pc=dict(engine='fixed_view_prior_v1',source_ids=tools,weights=[0.,.4,.6],code_revision=git_head());save_json(pp/'config.json',pc);results={}
for split in ('dev','confirm','val'):
 p=predict(np.stack([get_source(k,split) for k in tools],1),pc['weights']);np.save(pp/f'{split}_predictions.npy',p);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'))
save_json(pp/'metrics.json',dict(candidate=prior,metrics=results,status='evaluated'));print('PRIOR',results['val']['mse'],flush=True)
parent='20261105_p14structural_hullearly_mse';memory='20261105_variabletree_calendar_d2_mse';sources=['run:'+s for s in (parent,memory,prior)]
for alpha in (.1,.5,.75):
 name=f'20261105_calendarmemorystrength_a{int(alpha*100)}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=False);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(1800);cfg=dict(engine='masked_residual_strength_v1',source_ids=sources,columns=['MULL','LUFL'],reference_strength=.25,strength=alpha,code_revision=git_head());save_json(out/'config.json',cfg);results={}
 for split in ('dev','confirm','val'):
  args=[get_source(k,split) for k in sources];assert np.array_equal(args[0][:,:,[3,4]],args[1][:,:,[3,4]]),'Parent variable scope must equal memory expert';p=apply(*args,cfg);assert np.array_equal(p[:,:,[0,1,2,5,6]],args[0][:,:,[0,1,2,5,6]]);assert np.array_equal(apply(*args,dict(cfg,strength=.25)),args[0]);np.save(out/f'{split}_predictions.npy',p);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'))
 save_json(out/'metrics.json',dict(candidate=name,metrics=results,status='evaluated'));save_json(out/'contracts.json',dict(reference_strength_exact=True,other_five_columns_unchanged=True,parent_two_columns_equal_memory=True));print(name,results['val']['mse'],flush=True)
