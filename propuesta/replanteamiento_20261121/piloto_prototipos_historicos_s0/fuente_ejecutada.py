"""Train-only historical route prototypes; no query target used in retrieval/selection."""
from base_replanteo import *
import shutil,pandas as pd
from scipy.spatial import cKDTree
from numba import njit,prange,set_num_threads
@njit(cache=True,parallel=True)
def select_prototypes(query_xy,memory_xy,memory_geometry,neighbors,weights,default_shapes,default_valid):
 n,K=neighbors.shape;chosen=np.full((n,6),-1,np.int32);shape=np.zeros((n,6,14),np.float32)
 for q in prange(n):
  transformed=np.zeros((K,14),np.float64);D=np.zeros((K,K),np.float64);mind=np.full(K,np.inf)
  qdx=query_xy[q,2]-query_xy[q,0];qdy=query_xy[q,3]-query_xy[q,1];qnorm=max(np.sqrt(qdx*qdx+qdy*qdy),1e-9)
  for j in range(K):
   idx=neighbors[q,j];odx=memory_xy[idx,2]-memory_xy[idx,0];ody=memory_xy[idx,3]-memory_xy[idx,1];onorm=max(np.sqrt(odx*odx+ody*ody),1e-9)
   cos=(qdx*odx+qdy*ody)/(qnorm*onorm);sin=(qdy*odx-qdx*ody)/(qnorm*onorm);scale=qnorm/onorm
   for point in range(7):
    a=memory_geometry[idx,2*point];b=memory_geometry[idx,2*point+1]
    transformed[j,2*point]=scale*(cos*a-sin*b);transformed[j,2*point+1]=scale*(sin*a+cos*b)
   straight=0.
   for point in range(7):straight+=np.sqrt(transformed[j,2*point]**2+transformed[j,2*point+1]**2)/7
   mind[j]=straight
   for k in range(2):
    if not default_valid[q,k]:continue
    e=0.
    for point in range(7):
     a=transformed[j,2*point]-default_shapes[q,k,2*point];b=transformed[j,2*point+1]-default_shapes[q,k,2*point+1]
     e+=np.sqrt(a*a+b*b)/7
    mind[j]=min(mind[j],e)
  for j in range(K):
   for k in range(j):
    e=0.
    for point in range(7):
     a=transformed[j,2*point]-transformed[k,2*point];b=transformed[j,2*point+1]-transformed[k,2*point+1];e+=np.sqrt(a*a+b*b)/7
    D[j,k]=e;D[k,j]=e
  selected=np.zeros(K,np.uint8)
  for rank in range(6):
   winner=-1;best=0.
   for j in range(K):
    if weights[q,j]<=0 or selected[j]:continue
    gain=weights[q,j] if rank==0 else 0.
    if rank>0:
     for k in range(K):gain+=weights[q,k]*max(0.,mind[k]-D[j,k])
    if gain>best:best=gain;winner=j
   if winner<0:break
   selected[winner]=1;chosen[q,rank]=neighbors[q,winner];shape[q,rank]=transformed[winner]
   for j in range(K):mind[j]=min(mind[j],D[j,winner])
 return chosen,shape
def main():
 access=deny_test();start=time.time();set_num_threads(2);run=OUT/'piloto_prototipos_historicos_s0';assert not run.exists();run.mkdir()
 shutil.copy2(__file__,run/'fuente_ejecutada.py')
 protocol=dict(query=['o_lon','o_lat','d_lon','d_lat','timestamp'],scope='Train-only matched8candidate geometry coverage gate, not primaryMAE.',
 neighbors=64,historical_prototypes=6,source_UTCweek_fold_excluded=True,
 candidate_selection='Nearest64in4Dendpointkm;weights Gaussian250m,cyclic4hour and weekday/weekend match(0.35mismatch);500m4Dcutoff. First maximumweight, then greedy weighted geometric facility coverage against default routes/straight.',
 reanchoring='Rotate/scale historical relative shape by old/new OD chord. A latent geometric prior, not guaranteed drivable route.',
 comparison='Protected2publicroutes plus6historical prototypes andstraight, versus protected8publicroutes andstraight; same8routecount.',
 gate='At least10percent and25meters reduction versus the prior expanded8route geometry oracle.',
 query_true_geometry_used_only_after_selection=True,Validation_access=False,Test='sealed')
 write(run/'protocolo.json',protocol)
 tr=dict(np.load(DATA/'train_x.npz'));eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index'];gt=dict(np.load(DATA/'geometria_train.npz'))
 pos=np.searchsorted(gt['row'],tr['row'][eligible]);safe=np.minimum(pos,len(gt['row'])-1)
 valid=(pos<len(gt['row']))&(gt['row'][safe]==tr['row'][eligible])&(gt['X'][safe,-1]<=500)&np.isfinite(gt['X'][safe,1:15]).all(1)
 memory_train=eligible[valid];memory_geometry=gt['X'][safe[valid],1:15]
 def xy(d,ii):
  sx=111195*np.cos(np.deg2rad(41.16))/1000
  return np.column_stack([(d['o_lon'][ii]+8.62)*sx,(d['o_lat'][ii]-41.16)*111.195,(d['d_lon'][ii]+8.62)*sx,(d['d_lat'][ii]-41.16)*111.195])
 memory_xy=xy(tr,memory_train);memory_fold=(tr['timestamp'][memory_train]//604800)%5
 previous=dict(np.load(OUT/'piloto_proyeccion_incierta_s0_v2/predicciones_oraculo_Train.npz'))
 lookup=np.full(1710670,-1,np.int32);lookup[tr['row']]=np.arange(len(tr['row']));query_train=lookup[previous['row']];assert (query_train>=0).all()
 assert np.array_equal(tr['timestamp'][query_train],previous['timestamp'])
 query_xy=xy(tr,query_train);query_fold=(tr['timestamp'][query_train]//604800)%5
 current=np.load(OUT/'banco_proyeccion_incierta_s0/train_banco.npz')['bank'][query_train]
 defaults=current[:,:2,:14].astype(np.float32);default_valid=current[:,:2,20]>.5;del current
 local=pd.to_datetime(tr['timestamp'],unit='s',utc=True).tz_convert('Europe/Lisbon')
 hour=np.asarray(local.hour)+np.asarray(local.minute)/60+np.asarray(local.second)/3600;weekend=np.asarray(local.dayofweek)>=5
 memory_hour=hour[memory_train];memory_weekend=weekend[memory_train]
 selected=np.full((len(query_train),6),-1,np.int32);shapes=np.zeros((len(query_train),6,14),np.float32);nearest=np.empty(len(query_train),np.float32)
 for f in range(5):
  take=np.flatnonzero(query_fold==f);source=np.flatnonzero(memory_fold!=f)
  tree=cKDTree(memory_xy[source]);distance,index=tree.query(query_xy[take],k=64,workers=2);neighbors=source[index]
  clock=np.abs(memory_hour[neighbors]-hour[query_train[take],None]);clock=np.minimum(clock,24-clock)
  weights=np.exp(-.5*(distance/.25)**2-.5*(clock/4)**2)*np.where(memory_weekend[neighbors]==weekend[query_train[take],None],1.,.35)
  weights[distance>.5]=0
  chosen,G=select_prototypes(query_xy[take],memory_xy,memory_geometry,neighbors,weights,defaults[take],default_valid[take])
  selected[take]=chosen;shapes[take]=G;nearest[take]=distance[:,0]
  assert np.all(memory_fold[chosen[chosen>=0]]!=f)
  print(json.dumps(dict(stage='historical_geometry_gate',fold=f,queries=len(take),seconds=time.time()-start)),flush=True)
 qpos=np.searchsorted(gt['row'],tr['row'][query_train]);truth=gt['X'][qpos,1:15].reshape(-1,1,7,2)
 errors=np.sqrt(np.square(shapes.reshape(-1,6,7,2)-truth).sum(-1)).mean(-1)*1000;errors[selected<0]=np.inf
 historical=np.minimum(previous['geometry_baseline_m'],errors.min(1));old=previous['geometry_expanded_m']
 reduction=float(np.mean(old-historical));relative=reduction/float(old.mean());passed=reduction>=25 and relative>=.1
 source_rows=np.full_like(selected,-1);source_rows[selected>=0]=tr['row'][memory_train[selected[selected>=0]]]
 np.savez_compressed(run/'predicciones_oraculo_Train.npz',row=previous['row'],timestamp=previous['timestamp'],
  previous_public8_geometry_m=old,historical8_geometry_m=historical,selected_source_rows=source_rows,prototype_geometry_error_m=errors,
  nearest_OD4D_km=nearest,prototype_relative_shapes=shapes,available=selected>=0)
 result=dict(passed=bool(passed),n=len(query_train),memory_Train_trips=len(memory_train),
  mean_public8_geometry_m=float(old.mean()),mean_historical8_geometry_m=float(historical.mean()),mean_reduction_m=reduction,relative_reduction=relative,
  fraction_improving_more25m=float(np.mean(old-historical>25)),fraction_with_history=float(np.mean((selected>=0).any(1))),
  own_fold_never_retrieved=True,no_query_geometry_in_selector=True,Validation_access=False,Test_access=False,
  source_sha256=sha(run/'fuente_ejecutada.py'),predictions_sha256=sha(run/'predicciones_oraculo_Train.npz'),actual_data_files_accessed=sorted(access),seconds=time.time()-start,
  limitation='Reanchored historical latent shapes are not guaranteed road-connected. An oracle coverage gain alone is not a time-prediction gain.')
 write(run/'resultado.json',result);print(json.dumps({k:result[k] for k in ['passed','n','mean_public8_geometry_m','mean_historical8_geometry_m','mean_reduction_m','relative_reduction','fraction_with_history','seconds']}),flush=True)
if __name__=='__main__':main()
