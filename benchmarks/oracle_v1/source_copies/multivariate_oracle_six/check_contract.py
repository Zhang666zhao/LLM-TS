"""Synthetic GPU-only forward/backward checks, no optimizer or training run."""
import argparse
import json
from pathlib import Path
import torch
from conditioning import backbone,Forecast,objective,state_hash
from alignment import parameter_hash,historical_run

def main():
    p=argparse.ArgumentParser(); p.add_argument('--project',type=Path,required=True); p.add_argument('--method',required=True)
    p.add_argument('--dataset',required=True)
    a=p.parse_args(); torch.set_num_threads(4); torch.manual_seed(2021)
    c={'weather':21,'electricity':321}.get(a.dataset,7)
    net,cfg,_=backbone(a.project,a.method,a.dataset,c); original=parameter_hash(net)
    path=historical_run(a.project,a.method,a.dataset)/'initial_hash.json'
    if path.exists(): assert original==json.loads(path.read_text())['all']
    net=net.cuda().eval()
    x=torch.randn(2,512,c,device='cuda'); marks=torch.randn(2,512,5 if a.dataset in ('ETTm1','ETTm2','weather') else 4,device='cuda'); z=torch.zeros(2,c,dtype=torch.long,device='cuda')
    with torch.no_grad(): reference=net(x,marks,None,None) if a.method in ('itransformer','fredf') else net(x)
    model=Forecast(net,a.method,'A',c).cuda().eval()
    assert parameter_hash(net)==original
    with torch.no_grad(): torch.testing.assert_close(model(x,marks,z),reference,rtol=0,atol=0)
    with torch.no_grad():
        model.condition.net[-1].weight.normal_(0,.02)
        p0=model(x,marks,z); changed=z.clone(); changed[:,3]=1; p1=model(x,marks,changed)
        keep=[i for i in range(c) if i!=3]
        torch.testing.assert_close(p0[:,:,keep],p1[:,:,keep],rtol=0,atol=0)
        assert not torch.equal(p0[:,:,3],p1[:,:,3]),'Labels disconnected'
    model.train(); loss=objective(model(x,marks,z),torch.randn(2,64,c,device='cuda'),a.method,cfg); loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in net.parameters())
    assert model.condition.net[-1].weight.grad.abs().sum()>0
    # New pristine backbone avoids residual forward hooks from A.
    torch.manual_seed(2021); net2,_,_=backbone(a.project,a.method,a.dataset,c)
    frozen=Forecast(net2,a.method,'B',c).cuda(); frozen.train(); before=state_hash(net2)
    assert not net2.training and all(not p.requires_grad for p in net2.parameters())
    z2=z.clone(); z2[:,4]=-1
    with torch.no_grad():
        frozen.condition.net[-1].weight.normal_(0,.02)
        b0=frozen(x,marks,z); b1=frozen(x,marks,z2)
        keep=[i for i in range(c) if i!=4]
        torch.testing.assert_close(b0[:,:,keep],b1[:,:,keep],rtol=0,atol=0)
        assert not torch.equal(b0[:,:,4],b1[:,:,4])
    objective(frozen(x,marks,z),torch.randn(2,64,c,device='cuda'),a.method,cfg).backward()
    assert all(p.grad is None for p in net2.parameters()) and state_hash(net2)==before
    assert frozen.condition.net[-1].weight.grad.abs().sum()>0
    print(json.dumps(dict(dataset=a.dataset,method=a.method,status='passed',shape=[2,64,c],zero_head_equivalence=True,
                         label_channel_isolation=True,A_gradients=True,B_frozen_parameters_and_buffers=True,
                         historical_initial_hash_match=True if path.exists() else None)))

if __name__=='__main__': main()
