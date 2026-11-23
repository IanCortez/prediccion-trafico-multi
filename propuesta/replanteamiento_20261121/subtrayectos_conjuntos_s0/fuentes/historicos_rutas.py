"""Crossfit historical route geometry and time summaries from eligible Train only."""
from base_replanteo import *
import pandas as pd
from scipy.spatial import cKDTree
from numba import njit,prange,set_num_threads
from consulta_od import od_records
from piloto_prototipos_ruta import select_prototypes
HISTORY_NAMES=['shape_'+str(i) for i in range(14)]+['path_km','log_circuity','median_time600','MAD_time600','cluster_mass','mean_OD4D_km','mean_clock_difference12h','log1p_effective_historical_n','mean_shape_dispersion_km','available','mean_source_age_years','mean_log_ODscale']
@njit(cache=True)
def weighted_median(values,weights):
 order=np.argsort(values);total=weights.sum();cumulative=0.
 if total<=0:return 0.
 for idx in order:
  cumulative+=weights[idx]
  if cumulative>=.5*total:return values[idx]
 return values[order[-1]]
@njit(cache=True,parallel=True)
def summarize_history(qxy,qts,mxy,mgeometry,mlength,mtime,mts,neighbors,distance,clock,weights,chosen,shapes):
 n,K=neighbors.shape;H=np.zeros((n,6,26),np.float32)
 for q in prange(n):
  transform=np.zeros((K,14),np.float64);scales=np.ones(K,np.float64);labels=np.full(K,-1,np.int32)
  qdx=qxy[q,2]-qxy[q,0];qdy=qxy[q,3]-qxy[q,1];qnorm=max(np.sqrt(qdx*qdx+qdy*qdy),1e-9)
  for j in range(K):
   idx=neighbors[q,j];odx=mxy[idx,2]-mxy[idx,0];ody=mxy[idx,3]-mxy[idx,1];onorm=max(np.sqrt(odx*odx+ody*ody),1e-9)
   cos=(qdx*odx+qdy*ody)/(qnorm*onorm);sin=(qdy*odx-qdx*ody)/(qnorm*onorm);scales[j]=qnorm/onorm
   for p in range(7):
    a=mgeometry[idx,2*p];b=mgeometry[idx,2*p+1];transform[j,2*p]=scales[j]*(cos*a-sin*b);transform[j,2*p+1]=scales[j]*(sin*a+cos*b)
   best=np.inf
   for rank in range(6):
    if chosen[q,rank]<0:continue
    error=0.
    for p in range(7):
     a=transform[j,2*p]-shapes[q,rank,2*p];b=transform[j,2*p+1]-shapes[q,rank,2*p+1];error+=np.sqrt(a*a+b*b)/7
    if error<best:best=error;labels[j]=rank
  total=weights[q].sum()
  if total<=0:continue
  for rank in range(6):
   medoid=chosen[q,rank]
   if medoid<0:continue
   ww=np.zeros(K,np.float64);tt=np.zeros(K,np.float64);mass=0.;w2=0.;odsum=0.;clocksum=0.;dispersion=0.;agesum=0.;logscale=0.
   for j in range(K):
    if labels[j]!=rank:continue
    w=weights[q,j];idx=neighbors[q,j];ww[j]=w;tt[j]=mtime[idx]*scales[j];mass+=w;w2+=w*w
    odsum+=w*distance[q,j];clocksum+=w*clock[q,j]/12;agesum+=w*(qts[q]-mts[idx])/(365*86400);logscale+=w*np.log(scales[j])
    error=0.
    for p in range(7):
     a=transform[j,2*p]-shapes[q,rank,2*p];b=transform[j,2*p+1]-shapes[q,rank,2*p+1];error+=np.sqrt(a*a+b*b)/7
    dispersion+=w*error
   if mass<=0:continue
   median=weighted_median(tt,ww);mad=weighted_median(np.abs(tt-median),ww)
   odx=mxy[medoid,2]-mxy[medoid,0];ody=mxy[medoid,3]-mxy[medoid,1];ratio=qnorm/max(np.sqrt(odx*odx+ody*ody),1e-9)
   path=mlength[medoid]*ratio/1000
   H[q,rank,:14]=shapes[q,rank];H[q,rank,14]=path;H[q,rank,15]=np.log(max(path/qnorm,.01))
   H[q,rank,16]=median/600;H[q,rank,17]=mad/600;H[q,rank,18]=mass/total
   H[q,rank,19]=odsum/mass;H[q,rank,20]=clocksum/mass;H[q,rank,21]=np.log1p(mass*mass/max(w2,1e-20))
   H[q,rank,22]=dispersion/mass;H[q,rank,23]=1.;H[q,rank,24]=agesum/mass;H[q,rank,25]=logscale/mass
 return H
def query_xy(d):
 sx=111195*np.cos(np.deg2rad(41.16))/1000
 return np.column_stack([(d['o_lon']+8.62)*sx,(d['o_lat']-41.16)*111.195,(d['d_lon']+8.62)*sx,(d['d_lat']-41.16)*111.195])
def make_memory(path):
 tr=dict(np.load(DATA/'train_x.npz'));y=np.load(DATA/'train_y.npy');eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index'];gt=dict(np.load(DATA/'geometria_train.npz'))
 pos=np.searchsorted(gt['row'],tr['row'][eligible]);safe=np.minimum(pos,len(gt['row'])-1)
 valid=(pos<len(gt['row']))&(gt['row'][safe]==tr['row'][eligible])&(gt['X'][safe,-1]<=500)&np.isfinite(gt['X'][safe,1:15]).all(1)
 selected=eligible[valid];g=gt['X'][safe[valid]]
 d={k:tr[k][selected] for k in ['o_lon','o_lat','d_lon','d_lat','timestamp']};local=pd.to_datetime(d['timestamp'],unit='s',utc=True).tz_convert('Europe/Lisbon')
 assert np.all(d['timestamp']+y[selected]<read(DATA/'manifiesto.json')['validation']['start_timestamp'])
 np.savez_compressed(path,xy=query_xy(d),geometry=g[:,1:15],length_m=g[:,0],time_s=y[selected],timestamp=d['timestamp'],row=tr['row'][selected],train_index=selected,
  fold=((d['timestamp']//604800)%5).astype(np.int8),hour=np.asarray(local.hour)+np.asarray(local.minute)/60+np.asarray(local.second)/3600,weekend=np.asarray(local.dayofweek)>=5)
 return dict(n=len(selected),sha256=sha(path),bytes=path.stat().st_size,all_source_arrivals_before_Validation=True,all_memory_rows_in_eligible_Train=True)
class HistoryRouteBank:
 def __init__(self,path):
  self.memory=dict(np.load(path));self.trees={};set_num_threads(2)
 def _build(self,d,public_bank,training):
  m=self.memory;n=len(d['timestamp']);xy=query_xy(d)
  local=pd.to_datetime(d['timestamp'],unit='s',utc=True).tz_convert('Europe/Lisbon');hour=np.asarray(local.hour)+np.asarray(local.minute)/60+np.asarray(local.second)/3600;weekend=np.asarray(local.dayofweek)>=5
  fold=(d['timestamp']//604800)%5 if training else np.full(n,5)
  H=np.zeros((n,6,26),np.float32);sources=np.full((n,6),-1,np.int32)
  for f in np.unique(fold):
   take=np.flatnonzero(fold==f)
   if f not in self.trees:
    index=np.flatnonzero(m['fold']!=f) if f<5 else np.arange(len(m['row']))
    self.trees[int(f)]=(cKDTree(m['xy'][index]),index)
   tree,index=self.trees[int(f)];distance,near=tree.query(xy[take],k=64,workers=2);neighbors=index[near]
   clock=np.abs(m['hour'][neighbors]-hour[take,None]);clock=np.minimum(clock,24-clock)
   weights=np.exp(-.5*(distance/.25)**2-.5*(clock/4)**2)*np.where(m['weekend'][neighbors]==weekend[take,None],1.,.35);weights[distance>.5]=0
   chosen,G=select_prototypes(xy[take],m['xy'],m['geometry'],neighbors,weights,public_bank[take,:2,:14].astype(np.float32),public_bank[take,:2,20]>.5)
   H[take]=summarize_history(xy[take],d['timestamp'][take],m['xy'],m['geometry'],m['length_m'],m['time_s'],m['timestamp'],neighbors,distance,clock,weights,chosen,G)
   source_rows=np.full_like(chosen,-1);valid=chosen>=0;source_rows[valid]=m['row'][chosen[valid]];sources[take]=source_rows
   if f<5:assert np.all(m['fold'][chosen[valid]]!=f)
  assert np.isfinite(H).all()
  return H.astype(np.float16),sources
 def build(self,items,training=False):
  from rutas_proyeccion_incierta import build_uncertain_routes
  d=od_records(items);public,_=build_uncertain_routes(items,training=training)
  return self._build(d,public.astype(np.float16),training)
