"""Output-local conditional heads around unchanged official sources (no source edits)."""
import importlib
import importlib.util
import sys
from types import SimpleNamespace
import torch
from torch import nn
from alignment import load_historical_config, parameter_hash

METHODS = ('dlinear','patchtst','itransformer','frets','fredf')

def backbone(project, method):
    cfg, audit = load_historical_config(project,method,'exchange_rate')
    cfg['enc_in']=8
    if method=='dlinear':
        source=project/'third_party/LTSF-Linear/models/DLinear.py'
        spec=importlib.util.spec_from_file_location('oracle_dlinear',source)
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    else:
        repo={'patchtst':'PatchTST/PatchTST_supervised','frets':'FreTS','itransformer':'iTransformer','fredf':'iTransformer'}[method]
        sys.path.insert(0,str(project/'research/baselines_512_64/sources'/repo))
        name={'patchtst':'models.PatchTST','frets':'models.FreTS','itransformer':'model.iTransformer','fredf':'model.iTransformer'}[method]
        module=importlib.import_module(name)
    net=module.Model(SimpleNamespace(**cfg))
    audit.update(deliberate_differences=[], old_selection='all-variable-validation-MSE',
                 new_selection='all-variable-validation-MSE', claim='multivariate training config restored')
    return net,cfg,audit

class Conditional(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.embedding=nn.Embedding(3,8)
        self.net=nn.Sequential(nn.Linear(width+8,32),nn.GELU(),nn.Linear(32,64))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self,h,z):
        return self.net(torch.cat([h,self.embedding(z+1)],dim=-1))

class Forecast(nn.Module):
    def __init__(self,net,method,strategy):
        super().__init__(); self.backbone=net; self.method=method; self.strategy=strategy
        self.z=None; self.decomp=None
        # fork_rng keeps added-module initialization from changing backbone dropout RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2022)
            if strategy=='B': self.condition=Conditional(128)
            elif strategy=='A':
                if method=='dlinear': width=1024
                elif method=='patchtst':
                    assert not net.decomposition and not net.model.individual
                    width=net.model.head_nf
                elif method=='frets': width=net.fc[-1].in_features
                else: width=net.projector.in_features
                self.condition=Conditional(width)
        if strategy=='B': net.requires_grad_(False); net.eval()
        if strategy=='A':
            if method=='dlinear': net.decompsition.register_forward_hook(self._decomp)
            elif method=='patchtst': net.model.head.register_forward_hook(self._patch)
            elif method=='frets': net.fc[-1].register_forward_hook(self._linear)
            else: net.projector.register_forward_hook(self._linear)

    def _decomp(self,module,args,out): self.decomp=out
    def _patch(self,module,args,out):
        return out+self.condition(args[0].flatten(-2),self.z)
    def _linear(self,module,args,out):
        # iTransformer may have timestamp tokens after the eight variable tokens.
        delta=self.condition(args[0][:,:8],self.z)
        if out.shape[1]>8: delta=torch.cat([delta,torch.zeros_like(out[:,8:])],dim=1)
        return out+delta
    def train(self,mode=True):
        super().train(mode)
        if self.strategy=='B': self.backbone.eval()
        return self
    def forward(self,x,marks,z):
        if x.ndim!=3 or x.shape[1:]!=(512,8) or z.shape!=(len(x),8): raise ValueError('Input contract')
        if z.dtype!=torch.long or not ((z>=-1)&(z<=1)).all(): raise ValueError('Discrete labels required')
        self.z=z
        if self.method in ('itransformer','fredf'): p=self.backbone(x,marks,None,None)
        else: p=self.backbone(x)
        if self.strategy=='A' and self.method=='dlinear':
            seasonal,trend=self.decomp
            p=p+self.condition(torch.cat([seasonal.transpose(1,2),trend.transpose(1,2)],-1),z).transpose(1,2)
        elif self.strategy=='B':
            h=torch.cat([p.transpose(1,2),x[:,-64:].transpose(1,2)],-1)
            p=p+self.condition(h,z).transpose(1,2)
        self.z=None; self.decomp=None
        if p.shape!=(len(x),64,8): raise ValueError('Prediction contract')
        return p

def objective(p,y,method,cfg):
    assert p.shape==y.shape and p.shape[1:]==(64,8)
    mse=(p-y).square().mean()
    if method!='fredf': return mse
    return cfg['rec_lambda']*mse+cfg['auxi_lambda']*(torch.fft.rfft(p,dim=1)-torch.fft.rfft(y,dim=1)).abs().mean()

def state_hash(net):
    import hashlib
    h=hashlib.sha256()
    for n,t in net.state_dict().items(): h.update(n.encode()); h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
