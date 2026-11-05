"""Predict from a fresh CSV containing exactly 512 past observations."""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from forecast import COLS,L,load_forecaster

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--history',type=Path,required=True)
    parser.add_argument('--method',default='selected')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new output file')
    with args.history.open() as file:
        reader=csv.DictReader(file)
        if reader.fieldnames!=['date']+COLS:parser.error('Expected date and '+','.join(COLS))
        rows=list(reader)
    if len(rows)!=L:parser.error('Supply exactly 512 historical rows')
    x=np.array([[float(r[k]) for k in COLS] for r in rows]);t=np.array([r['date'] for r in rows],dtype='datetime64[m]')
    torch.set_num_threads(4)
    model=load_forecaster(args.plan)
    mid=model.plan['selected'] if args.method=='selected' else args.method
    if mid not in model.plan['outputs']:parser.error('Unknown method: '+mid)
    p=model.predict(x,t,mid)
    future=t[-1]+np.arange(1,len(p)+1)*np.timedelta64(model.plan['minutes_per_step'],'m')
    with args.output.open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(['date']+COLS)
        writer.writerows([[str(ti),*yi.tolist()] for ti,yi in zip(future,p)])
