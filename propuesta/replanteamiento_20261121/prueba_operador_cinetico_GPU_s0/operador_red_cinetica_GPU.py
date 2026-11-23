"""Canonical64 Float64 sparseforward; identical finite15s operator as CPU."""
import numpy as np,pandas as pd,torch
from collections import OrderedDict
from coordenadas_trayecto_2d import metric_queries
class CudaRoadClock:
 def __init__(self,model,batch_size=64,device='cuda',cache_limit=4):
  assert batch_size==64 and cache_limit==4
  self.model=model;self.batch_size=batch_size;self.device=torch.device(device)
  self.cache=OrderedDict();self.cache_limit=cache_limit;self.cache_peak=0
 def kernel(self,hour):
  hour=int(hour)
  if hour in self.cache:
   self.cache.move_to_end(hour);return self.cache[hour]
  QT,stop,prior=self.model.kernel(hour)
  value=(torch.sparse_csr_tensor(torch.as_tensor(QT.indptr.astype(np.int64),device=self.device),
    torch.as_tensor(QT.indices.astype(np.int64),device=self.device),
    torch.as_tensor(QT.data,device=self.device),size=QT.shape,dtype=torch.float64,device=self.device),
    torch.as_tensor(stop,device=self.device),prior)
  # Retire before inserting the nextmatrix to keep the persistentLRU bounded.
  if len(self.cache)>=self.cache_limit:self.cache.popitem(last=False)
  self.cache[hour]=value;self.cache_peak=max(self.cache_peak,len(self.cache))
  return value
 @torch.inference_mode()
 def predict(self,queries,progress=None):
  xy=metric_queries(queries);B=len(xy);N=self.model.size;C=self.batch_size
  timestamp=np.asarray(queries['timestamp'],np.int64)
  clock=timestamp[:,None]+15*np.arange(120)[None]
  local=pd.to_datetime(clock.ravel(),unit='s',utc=True).tz_convert('Europe/Lisbon')
  hours=(np.asarray(local.dayofweek)*24+np.asarray(local.hour)).reshape(B,120)
  weights=np.zeros((B,116),np.float64);diagnostics=[None]*B
  # Grouping is solely by publicknownclock. Each column has its own fullfixed119tick horizon.
  for initial_hour in np.unique(hours[:,0]):
   indices=np.flatnonzero(hours[:,0]==initial_hour)
   for offset in range(0,len(indices),C):
    ii=indices[offset:offset+C];m=len(ii);h=hours[ii]
    origin=np.zeros((N,C),np.float64);destination=np.zeros_like(origin)
    _,_,prior=self.model.kernel(initial_hour)
    for j,i in enumerate(ii):
     oi,ow=self.model.measurement(xy[i,:2]);di,dw=self.model.measurement(xy[i,2:])
     iw=prior[oi]*ow;normalizer=float(iw.sum())
     diagnostics[i]=dict(origin_nodes=len(oi),destination_nodes=len(di),initial_normalizer=normalizer)
     if normalizer>0:origin[oi,j]=iw/normalizer
     destination[di,j]=dw
    flow=torch.as_tensor(origin,device=self.device);LD=torch.as_tensor(destination,device=self.device)
    result=torch.zeros((116,C),dtype=torch.float64,device=self.device)
    for n in range(120):
     new_flow=torch.empty_like(flow) if n<119 else None
     for hour in np.unique(h[:,n]):
      QT,stop,_=self.kernel(hour)
      columns=torch.as_tensor(np.flatnonzero(h[:,n]==hour),device=self.device)
      if n>=4:
       terminal=(flow*LD*stop[:,None]).sum(dim=0)
       result[n-4,columns]=terminal[columns]
      if n<119:
       advanced=torch.sparse.mm(QT,flow)
       new_flow[:,columns]=advanced[:,columns]
     # Inactivepadding columns remainzero, independent of partners/order.
     if n<119:
      new_flow[:,m:]=0.;flow=new_flow
    weights[ii]=result[:,:m].T.cpu().numpy()
    if progress:progress(dict(completed=min(offset+C,len(indices)),hour=int(initial_hour),total_hourqueries=len(indices)))
  support=weights.sum(1);positive=support>0
  probability=np.divide(weights,support[:,None],out=np.zeros_like(weights),where=support[:,None]>0)
  assert np.isfinite(weights).all() and (weights>=0).all() and (support<=1+1e-10).all()
  return dict(weights=weights,probability=probability,supported=positive,support=support,diagnostics=diagnostics)
