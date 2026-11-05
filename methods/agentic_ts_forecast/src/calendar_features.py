"""Record-calendar covariates; no weather, holidays, location or future sensor values."""
import numpy as np
import pandas as pd
def cov(d,mode):
 assert mode in ('hour','hourweek','houryear','hourweekyear');d=pd.DatetimeIndex(d);h=np.asarray(d.hour,dtype=float);a={'hour_sin':np.sin(2*np.pi*h/24).astype(np.float32),'hour_cos':np.cos(2*np.pi*h/24).astype(np.float32)}
 if mode in ('hourweek','hourweekyear'):
  w=np.asarray(d.dayofweek,dtype=float)+h/24;a.update(week_sin=np.sin(2*np.pi*w/7).astype(np.float32),week_cos=np.cos(2*np.pi*w/7).astype(np.float32))
 if mode in ('houryear','hourweekyear'):
  phase=(np.asarray(d.dayofyear,dtype=float)-1+h/24)/np.where(d.is_leap_year,366.,365.);a.update(year_sin=np.sin(2*np.pi*phase).astype(np.float32),year_cos=np.cos(2*np.pi*phase).astype(np.float32))
 return a
