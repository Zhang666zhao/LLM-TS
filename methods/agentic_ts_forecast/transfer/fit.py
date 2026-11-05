"""Single fixed fit on train only; pre-existing DLinear weights are the matched baseline."""
import argparse,csv,time,platform
from forecast import *
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--csv',required=True);ap.add_argument('--baseline',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--device',default='cuda:0');args=ap.parse_args()
    started=time.time();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'tools.npz').exists():raise FileExistsError(out/'tools.npz')
    cfg=read(Path(__file__).parent/'contract.json')
    if digest(args.csv)!=cfg['sha256']:raise ValueError('ETTm1 data version mismatch')
    with open(args.csv) as f:
        r=csv.DictReader(f);rows=[next(r) for _ in range(cfg['train_end'])]
    raw=np.array([[float(row[k]) for k in COLS] for row in rows]);minutes=stamp([row['date'] for row in rows])
    if not np.all(np.diff(minutes)==15):raise ValueError('unexpected cadence')
    mean,std=raw.mean(0),raw.std(0);torch.manual_seed(2021);device=args.device
    z=tensor((raw-mean)/std,device);tm=torch.as_tensor(minutes,device=device);period=cfg['steps_per_day']
    idx=torch.arange(-L,0,device=device);lead=torch.arange(H,device=device)
    def xy(orig):
        ti=torch.as_tensor(orig,device=device);return z[ti[:,None]+idx],z[ti[:,None]+lead],tm[ti]
    bank_t=np.arange(L,len(z)-H+1,cfg['bank_stride']);xb,_,bt=xy(bank_t);desc=descriptor(xb)
    dm=desc.mean(0);ds=desc.std(0,unbiased=False).clamp_min(.001)
    state=dict(bank_history=xb.cpu().numpy(),desc_mean=dm.cpu().numpy(),desc_scale=ds.cpu().numpy(),
               bank_desc=((desc-dm)/ds).cpu().numpy(),bank_minutes=bt.cpu().numpy())
    bank=ToolBank(state,period,device)
    dl=torch.load(args.baseline,map_location='cpu',weights_only=True)
    for tag,k in [('sw','Seasonal.weight'),('sb','Seasonal.bias'),('tw','Trend.weight'),('tb','Trend.bias')]:
        state[tag]=dl['backbone.Linear_'+k].numpy()
    for kind,penalty in [('ridge_full',.10),('ridge_clock',.03),('ridge_cross',.03),('A',.08)]:
        gram=rh=None;count=0
        for origins in batches(np.arange(2*L+H if kind=='A' else L,len(z)-H+1,cfg['fit_stride']),256):
            x,y,t=xy(origins);phase=(t%1440)//cfg['minutes_per_step']
            old=bank.older(x,t) if kind=='A' else None
            feat,center=features(x,phase,kind,period,old)
            a=feat.transpose(0,1).double();b=(y-center).permute(2,0,1).double()
            g=a.transpose(1,2)@a;h=a.transpose(1,2)@b
            if gram is None:gram,rh=g,h
            else:gram+=g;rh+=h
            count+=len(origins)
        reg=torch.eye(gram.shape[-1],device=device,dtype=torch.float64)*penalty*count;reg[-1,-1]=1e-8
        state[kind]=torch.linalg.solve(gram+reg,rh).float().cpu().numpy()
    w=torch.zeros(D,6,3,device=device).normal_(std=.08).requires_grad_()
    b=torch.zeros(D,3,device=device,requires_grad=True)
    leaf=torch.zeros(D,4,period,device=device).normal_(std=.01).requires_grad_()
    opt=torch.optim.Adam([w,b,leaf],lr=.015)
    for _ in range(180):
        t=torch.randint(L,len(z)-H+1,(96,),device=device);x,y,minutes_t=xy(t);phase=(minutes_t%1440)//15
        pred=tree_output(x,phase,period,w,b,leaf)
        opt.zero_grad();(pred-y).square().mean().backward();opt.step()
    for key,val in [('tree_w',w),('tree_b',b),('tree_leaf',leaf)]:state[key]=val.detach().cpu().numpy()
    if not all(np.isfinite(a).all() for a in state.values()):raise ValueError('nonfinite fitted state')
    np.savez_compressed(out/'tools.npz',**state)
    write(out/'fit.json',dict(contract=cfg,mean=mean.tolist(),std=std.tolist(),device=device,
          gpu=torch.cuda.get_device_name(),torch=torch.__version__,python=platform.python_version(),
          source_sha256=cfg['sha256'],dlinear_sha256=digest(args.baseline),state_sha256=digest(out/'tools.npz'),
          rows_seen=len(z),tree_steps=180,bank_size=len(bank_t),elapsed_s=time.time()-started))
    print('TRAIN_COMPLETE',len(z),len(bank_t),time.time()-started,flush=True)
if __name__=='__main__':main()
