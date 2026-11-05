"""ETT transfer pilot v1: history-only tools and frozen numerical composition."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from torch.nn import functional as F

L,H,D=512,64,7
COLS=['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']
TOOLS=['DLinear','ridge_full','ridge_clock','ridge_cross','day','day_mean','A','R']
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False))
def read(path):return json.loads(Path(path).read_text())
def stamp(t):return np.asarray(t,dtype='datetime64[m]').astype(np.int64)
def batches(ts,size=128):
    for i in range(0,len(ts),size):yield ts[i:i+size]
def tensor(v,device):return torch.as_tensor(v,dtype=torch.float32,device=device)

def descriptor(x):
    centered=x-x[:,-96:].mean(1,keepdim=True)
    return torch.cat([centered.reshape(-1,32,16,D).mean(2).flatten(1),x[:,-24:].mean(1),
                      x[:,-96:].mean(1),x.mean(1),x.std(1,unbiased=False),
                      x[:,-24:].mean(1)-x[:,-48:-24].mean(1)],1)

def calendar(x,period):
    return torch.stack([x[:,L-k*period:L-k*period+H] for k in range(1,L//period+1)]).mean(0)

def features(x,phase,kind,period,old=None):
    center=x[:,-96:].mean(1,keepdim=True)
    z=x-center
    if kind=='ridge_full':items=[z]
    else:
        items=[z.reshape(-1,32,16,D).mean(2),z[:,-96:],calendar(z,period)]
        ang=phase*2*np.pi/period
        clock=torch.stack([ang.sin(),ang.cos(),(2*ang).sin(),(2*ang).cos()],1)
        items.append(clock[:,:,None].expand(-1,-1,D))
        if kind=='ridge_cross':
            summaries=torch.cat([z[:,-16:].mean(1),z[:,-96:].mean(1),x.std(1,unbiased=False)],1)
            items.append(summaries[:,:,None].expand(-1,-1,D))
        if kind=='A':
            oz=old-old[:,-96:].mean(1,keepdim=True)
            items.extend([oz.reshape(-1,32,16,D).mean(2), calendar(oz,period)])
    f=torch.cat(items,1).transpose(1,2)
    return torch.cat([f,torch.ones_like(f[:,:,:1])],2),center

def tree_input(x,phase,period):
    a=phase*2*np.pi/period
    unit=torch.ones_like(x[:,0,:])
    return torch.stack([x[:,-16:].mean(1)-x[:,-96:].mean(1),x[:,-96:].mean(1)-x.mean(1),
                        x[:,-96:].std(1,unbiased=False),x[:,-8:].mean(1)-x[:,-16:-8].mean(1),
                        unit*a.sin()[:,None],unit*a.cos()[:,None]],2)

def tree_output(x,phase,period,w,b,leaf):
    gate=torch.einsum('ndf,dfg->ndg',tree_input(x,phase,period),w).add(b).sigmoid()
    p,q,r=gate.unbind(2)
    mass=torch.stack([(1-p)*(1-q),(1-p)*q,p*(1-r),p*r],2)
    templates=torch.einsum('ndk,dkp->ndp',mass,leaf)
    slots=(phase.long()[:,None]+torch.arange(H,device=x.device))%period
    corr=templates.gather(2,slots[:,None,:].expand(-1,D,-1)).transpose(1,2)
    return calendar(x,period)+.25*x.std(1,unbiased=False).clamp_min(.05)[:,None,:]*corr.tanh()

class ToolBank:
    def __init__(self,state,period=96,device='cpu'):
        self.state={k:tensor(v,device) for k,v in state.items() if k!='bank_minutes'}
        self.minutes=torch.as_tensor(state['bank_minutes'],dtype=torch.int64,device=device)
        self.period=period
        self.device=device

    def older(self,x,minutes):
        s=self.state
        query=(descriptor(x)-s['desc_mean'])/s['desc_scale']
        distances=torch.cdist(query,s['bank_desc'])
        # The end of the old 64-step segment precedes the start of the current history.
        valid=self.minutes[None,:]+H*(1440//self.period)<=minutes[:,None]-L*(1440//self.period)
        if not valid.any(1).all():raise ValueError('No strictly earlier training case for this forecast origin')
        ids=distances.masked_fill(~valid,float('inf')).argmin(1)
        return s['bank_history'][ids]

    def forward(self,x,minutes):
        period=self.period;s=self.state
        phase=(minutes%(24*60))//(1440//period)
        padded=F.pad(x.transpose(1,2),(12,12),mode='replicate')
        trend=F.avg_pool1d(padded,25,stride=1)
        seasonal=x.transpose(1,2)-trend
        dl=(F.linear(seasonal,s['sw'],s['sb'])+F.linear(trend,s['tw'],s['tb'])).transpose(1,2)
        out={'DLinear':dl,'day':x[:,L-period:L-period+H], 'day_mean':calendar(x,period)}
        old=self.older(x,minutes)
        for k in ['ridge_full','ridge_clock','ridge_cross','A']:
            f,c=features(x,phase,k,period,old)
            out[k]=torch.einsum('ndf,dfh->nhd',f,s[k])+c
        out['R']=tree_output(x,phase,period,s['tree_w'],s['tree_b'],s['tree_leaf'])
        return {k:v.detach().cpu().numpy() for k,v in out.items()}

def compose(raw,nodes):
    preds=dict(raw)
    for name,cfg in nodes.items():
        kind=cfg['kind']
        if kind=='mean':
            preds[name]=(preds[cfg['sources'][0]]*.5+preds[cfg['sources'][1]]*.5).astype('float32')
        elif kind=='select':
            b=cfg['block'];out=np.empty_like(next(iter(preds.values())))
            for j,choices in enumerate(cfg['map']):
                for q,src in enumerate(choices):
                    out[:,q*b:(q+1)*b,j]=preds[src][:,q*b:(q+1)*b,j]
            preds[name]=out
        elif kind=='mix':
            out=preds[cfg['parent']].copy();j=COLS.index(cfg['variable']);w=cfg['alpha']
            out[:,:,j]=(1-w)*out[:,:,j].astype('float64')+w*preds[cfg['source']][:,:,j].astype('float64')
            preds[name]=out
        else:raise ValueError(kind)
    return preds

class Forecaster:
    def __init__(self,plan_path,device='cpu'):
        self.path=Path(plan_path).resolve()
        seal=read(self.path.parent/'freeze.json')
        if digest(self.path)!=seal['plan_sha256']:raise ValueError('Plan checksum mismatch')
        self.plan=read(self.path)
        if digest(Path(__file__))!=self.plan['runtime_sha256']:raise ValueError('Frozen runtime has changed')
        p=self.path.parent/self.plan['state']
        if digest(p)!=self.plan['state_sha256']:raise ValueError('Tool state checksum mismatch')
        with np.load(p) as s:self.tools=ToolBank(dict(s),self.plan['steps_per_day'],device)
        self.device=device

    def predict_batch(self,history,timestamps,method=None,standardized=False):
        x=np.asarray(history,dtype='float64');t=stamp(timestamps)
        if x.ndim!=3 or x.shape[1:]!=(L,D) or t.shape!=x.shape[:2] or not np.isfinite(x).all():
            raise ValueError('Expected finite [batch,512,7] observations and [batch,512] timestamps')
        if not np.all(np.diff(t,axis=1)==self.plan['minutes_per_step']):raise ValueError('Timestamp cadence mismatch')
        mu=np.asarray(self.plan['mean']);std=np.asarray(self.plan['std'])
        z=tensor((x-mu)/std,self.device);origin=torch.as_tensor(t[:,-1]+self.plan['minutes_per_step'],device=self.device)
        with torch.no_grad():p=compose(self.tools.forward(z,origin),self.plan['nodes'])
        selected=[method] if method else list(self.plan['outputs'])
        return {k:(p[self.plan['outputs'][k]] if standardized else p[self.plan['outputs'][k]]*std+mu) for k in selected}

    def predict(self,history,timestamps,method='M1'):
        return self.predict_batch(np.asarray(history)[None],np.asarray(timestamps)[None],method)[method][0]

def load_forecaster(plan_path,device='cpu'):return Forecaster(plan_path,device)
