"""Publicquery features separate from fit-only supervisedGPS targetstream; no newlargecache."""
from base_replanteo import *
import torch,gc
from inferencia import feature_input
from coordenadas_trayecto_2d import metric_queries
class BridgeInputs:
 def __init__(self,run):
  start=time.time()
  folder=OUT/'indices_escala_Train_s0';meta=read(folder/'resultado.json')
  assert meta['complete'] and meta['fit_n']==951718 and meta['held_n']==10000
  with np.load(folder/'metadatos_Train.npz') as z:
   self.index=z['train_index'];self.row=z['row'];self.timestamp=z['timestamp'];self.nfit=int(z['fit_n'])
  self.n=len(self.index);self.cutoff=meta['source_cutoff']
  assert self.n==self.nfit+10000 and len(np.unique(self.index))==self.n
  d,X,names=feature_input('train',read(BEST/'resultado.json'))
  keep=np.array([not name.startswith('mm_') for name in names])
  self.names=[name for name,k in zip(names,keep) if k];assert len(self.names)==206
  X=np.ascontiguousarray(X[self.index][:,keep]);assert np.isfinite(X).all()
  assert np.array_equal(d['row'][self.index],self.row) and np.array_equal(d['timestamp'][self.index],self.timestamp)
  self.query={key:d[key][self.index] for key in ['o_lon','o_lat','d_lon','d_lat','timestamp']}
  self.query_xy=metric_queries(self.query)
  y=np.load(DATA/'train_y.npy');self.truth=y[self.index].astype(np.float32)
  assert np.all(self.timestamp[:self.nfit]+self.truth[:self.nfit]<self.cutoff)
  assert np.all((self.truth>=60)&(self.truth<=1785)&(self.truth%15==0))
  with np.load(OUT/'oraculo_tiempo_calle_train_s0/predicciones_held_Train.npz') as z:
   for key,value in [('row',self.row[self.nfit:]),('timestamp',self.timestamp[self.nfit:]),('true',self.truth[self.nfit:])]:
    assert np.array_equal(z[key],value)
  mean=X[:self.nfit].mean(0,dtype=np.float64);std=X[:self.nfit].std(0,dtype=np.float64);std[std<1e-4]=1
  anchor_names=['network_fastest_time_600s','projected_length_freeflow_600s','projected_free_freeflow_600s','od_km']
  indices=np.repeat([self.names.index(n) for n in anchor_names],2);factors=np.repeat([600.,600.,600.,1000.],2)
  coef=[];logy=np.log(self.truth[:self.nfit].astype(float))
  for index,factor in zip(indices,factors):
   value=np.log(np.maximum(.1,X[:self.nfit,index].astype(float)*factor))
   coef.append(np.linalg.lstsq(np.column_stack([np.ones(self.nfit),value]),logy,rcond=None)[0])
  self.constructor=dict(xmean=mean.astype(np.float32),xstd=std.astype(np.float32),anchor_indices=indices,
   anchor_scale=factors.astype(np.float32),anchor_coef=np.asarray(coef,np.float32))
  self.x=torch.tensor(X,device='cuda');self.target_n=torch.tensor((self.truth[:self.nfit]/15).astype(np.int64),device='cuda')
  # Originalbank contains allTrain. Only the explicit originalfitindex whitelist can enter an auxiliaryobjective.
  bank=OUT/'memoria_puntos_Train_s0/puntos_Train.npz'
  assert sha(bank)==read(bank.parent/'resultado.json')['outputs_sha256']
  with np.load(bank) as z:
   assert np.array_equal(z['row'][self.index[:self.nfit]],self.row[:self.nfit])
   source_n=z['parent_gps_n'][self.index[:self.nfit]].astype(int)
   assert np.array_equal((source_n-1)*15,self.truth[:self.nfit])
   assert np.all(z['timestamp'][self.index[:self.nfit]]+z['time_s'][self.index[:self.nfit]]<self.cutoff)
   xy=z['xy_m'];offset=z['offsets']
   source_offset=offset[self.index[:self.nfit]]
   first=xy[source_offset].astype(float);last=xy[source_offset+source_n-1].astype(float)
   endpoint_error=max(float(np.max(np.linalg.norm(first-self.query_xy[:self.nfit,:2],axis=1))),
    float(np.max(np.linalg.norm(last-self.query_xy[:self.nfit,2:],axis=1))))
   assert endpoint_error<.005,endpoint_error
   self.private_xy=torch.tensor(xy,device='cuda')
   self.private_offset=torch.tensor(source_offset,device='cuda')
  self.fit_xy=torch.tensor(self.query_xy[:self.nfit],device='cuda',dtype=torch.float64)
  self.step=torch.arange(120,device='cuda')
  self.metadata=dict(constructor=self.constructor,features=self.names,fit_n=self.nfit,held_n=10000,source_cutoff=self.cutoff,
   fit_metadata_sha256=sha(folder/'metadatos_Train.npz'),source_pointbank_sha256=sha(bank))
  torch.save(self.metadata,run/'constructor.pt')
  write(run/'verificacion_datos.json',dict(passed=True,fit_n=self.nfit,held_n=10000,
   all_fit_bank_rows_labels_and_sourcearrivals_verified=True,all_fit_endpoint_maxerror_m=endpoint_error,
   public206_normalizers_and_anchors_fit951718only=True,no_new_largecache=True,
   original_bank_contains_allTrain_but_only_fit_whitelist_enters_auxiliary=True,
   prediction_function_has_no_GPS_or_trueN_argument=True,
   resident_MiB=torch.cuda.memory_allocated()/1024**2,constructor_sha256=sha(run/'constructor.pt'),seconds=time.time()-start))
  del X,d,y,xy,source_n,offset,first,last,logy;gc.collect()
  print(json.dumps(dict(stage='bridge_all_data_ready',resident_MiB=torch.cuda.memory_allocated()/1024**2,seconds=time.time()-start)),flush=True)
 @torch.no_grad()
 def fit_targets(self,fit_ids,mode):
  assert mode in ['linear','observed']
  ids=torch.as_tensor(fit_ids,device='cuda',dtype=torch.long)
  assert ids.ndim==1 and bool(((ids>=0)&(ids<self.nfit)).all()),'AuxiliaryGPS only for whitelistedfitids'
  n=self.target_n[ids]
  if mode=='linear':
   positions=torch.zeros((len(ids),120,2),device='cuda',dtype=torch.float64)
   positions[...,0]=(self.step[None]/n[:,None]).clamp_max(1)
  else:
   indices=self.private_offset[ids,None]+torch.minimum(self.step[None],n[:,None])
   points=self.private_xy[indices].double()
   frame=self.fit_xy[ids];delta=frame[:,2:]-frame[:,:2];den=delta.square().sum(-1)
   relative=points-frame[:,None,:2]
   a=(relative*delta[:,None]).sum(-1)/den[:,None]
   b=(delta[:,None,0]*relative[...,1]-delta[:,None,1]*relative[...,0])/den[:,None]
   positions=torch.stack([a,b],-1)
   positions[:,0]=0
   endpoint=torch.tensor([1.,0.],device='cuda',dtype=torch.float64)
   positions=torch.where((self.step[None]>=n[:,None])[...,None],endpoint,positions)
  positions=positions.float()
  return torch.diff(positions,dim=1),n
