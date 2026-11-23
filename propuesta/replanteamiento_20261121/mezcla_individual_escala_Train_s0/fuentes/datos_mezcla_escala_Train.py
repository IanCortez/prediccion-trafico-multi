"""GPU stream over compactreferences; never cache full-population rawtokens."""
from base_replanteo import *
import torch,gc
from inferencia import feature_input
from tokens_subtrayectos import query_parts_sub
from tokens_transferencia_arco import ArcTransferBank,add_public_compatibility,compatibility_indices
class StreamInputs:
 def __init__(self,run):
  start=time.time();folder=OUT/'indices_escala_Train_s0';r=read(folder/'resultado.json')
  assert r['complete'] and r['fit_n']==951718 and r['held_n']==10000
  assert sha(folder/'indices_fuentes.npz')==r['indices_sha256']
  assert r['held_all10000_refs_identical_to_smallpilot'] and r['wholeparentfold_exclusion_before_tree']
  with np.load(folder/'indices_fuentes.npz') as z:
   self.index=z['train_index'];self.row=z['row'];self.timestamp=z['timestamp']
   p=z['parent'];first=z['first'];last=z['last'];self.nfit=int(z['fit_n'])
  self.n=len(self.index);self.cutoff=r['source_cutoff']
  d,X,names=feature_input('train',read(BEST/'resultado.json'));keep=np.array([not name.startswith('mm_') for name in names])
  self.names=[name for name,k in zip(names,keep) if k];assert len(self.names)==206
  X=np.ascontiguousarray(X[self.index][:,keep])
  assert np.array_equal(d['row'][self.index],self.row) and np.array_equal(d['timestamp'][self.index],self.timestamp)
  all_y=np.load(DATA/'train_y.npy');self.truth=np.asarray(all_y[self.index],np.float32)
  assert np.all(self.timestamp[:self.nfit]+self.truth[:self.nfit]<self.cutoff)
  assert np.all((self.truth>=60)&(self.truth<=1785)&(self.truth%15==0))
  with np.load(OUT/'oraculo_tiempo_calle_train_s0/auxiliares_Train.npz') as old:
   assert np.array_equal(old['row'][40000:],self.row[self.nfit:]) and np.array_equal(old['true'][40000:],self.truth[self.nfit:])
  self.queries={key:d[key][self.index] for key in ['o_lon','o_lat','d_lon','d_lat','timestamp']}
  self.shape_indices,self.available_indices=compatibility_indices(self.names)
  parts=query_parts_sub(self.queries)
  self.x=torch.tensor(X,device='cuda');self.xy=torch.tensor(parts['xy_m'],device='cuda')
  self.hour=torch.tensor(parts['hour'],device='cuda');self.weekend=torch.tensor(parts['weekend'],device='cuda')
  self.parent=torch.tensor(p,device='cuda');self.first=torch.tensor(first,device='cuda');self.last=torch.tensor(last,device='cuda')
  self.y=torch.tensor(self.truth[:self.nfit],device='cuda')
  del p,first,last,parts,d,all_y;gc.collect()
  self.bank=ArcTransferBank('cuda')
  # Publicnormalization and allphysicaltimeanchors use only the fullprecutofffit labels.
  mean=X[:self.nfit].mean(0,dtype=np.float64);std=X[:self.nfit].std(0,dtype=np.float64);std[std<1e-4]=1
  names_phys=['network_fastest_time_600s','projected_length_freeflow_600s','projected_free_freeflow_600s','od_km']
  indices=np.repeat([self.names.index(name) for name in names_phys],2);scale=np.repeat([600.,600.,600.,1000.],2)
  coef=[];logy=np.log(self.truth[:self.nfit].astype(float))
  for index,factor in zip(indices,scale):
   value=np.log(np.maximum(.1,X[:self.nfit,index].astype(float)*factor))
   coef.append(np.linalg.lstsq(np.column_stack([np.ones(self.nfit),value]),logy,rcond=None)[0])
  self.common=dict(xmean=mean.astype(np.float32),xstd=std.astype(np.float32),anchor_indices=indices,
   anchor_scale=scale.astype(np.float32),anchor_coef=np.asarray(coef,np.float32))
  del X,logy;gc.collect()
  selected=np.random.default_rng(20261127).choice(self.nfit,32768,replace=False)
  totals={key:dict(count=0,total=np.zeros(38),square=np.zeros(38)) for key in ['observed','geometry']}
  immutable=[i for i in range(38) if i not in [16,32]]
  for a in range(0,len(selected),256):
   ids=torch.tensor(selected[a:a+256],device='cuda')
   _,obs=self.batch(ids,'observed');_,geo=self.batch(ids,'geometry')
   assert torch.equal(obs[...,immutable],geo[...,immutable])
   for key,raw in [('observed',obs),('geometry',geo)]:
    values=raw.float().cpu().numpy().reshape(-1,38);values=values[values[:,27]>.5].astype(float)
    v=totals[key];v['count']+=len(values);v['total']+=values.sum(0);v['square']+=(values**2).sum(0)
  self.norm={}
  for key,value in totals.items():
   m=value['total']/value['count'];s=np.sqrt(np.maximum(0,value['square']/value['count']-m*m));s[s<1e-4]=1
   self.norm[key]=dict(mean=m.astype(np.float32),std=s.astype(np.float32),fit_source_tokens=value['count'])
  assert np.array_equal(self.norm['observed']['mean'][immutable],self.norm['geometry']['mean'][immutable])
  assert np.array_equal(self.norm['observed']['std'][immutable],self.norm['geometry']['std'][immutable])
  self.metadata=dict(common=self.common,source_normalization=self.norm,features=self.names,shape_indices=self.shape_indices,
   available_indices=self.available_indices,fit_n=self.nfit,held_n=10000,source_cutoff=self.cutoff,sourceindices_sha256=r['indices_sha256'])
  torch.save(self.metadata,run/'constructor.pt')
  write(run/'verificacion_datos.json',dict(passed=True,fit_n=self.nfit,held_n=10000,
   sourceindices_sha256=r['indices_sha256'],held_rows_labels_identical_to_originalpilot=True,
   fit_publicnormalizers_and_anchors_fit_all_preheld_only=True,source_normalizers_query_sample_n=32768,source_normalizer_seed=20261127,
   sample_source_tokens=self.norm['observed']['fit_source_tokens'],source_control_only_changes_columns=[16,32],
   persistent_full_rawtokens=False,GPU_rawtokens_materialized_population=False,
   resident_input_MiB=torch.cuda.memory_allocated()/1024**2,constructor_sha256=sha(run/'constructor.pt'),seconds=time.time()-start))
  print(json.dumps(dict(stage='stream_inputs_ready',fit_n=self.nfit,held_n=10000,resident_MiB=torch.cuda.memory_allocated()/1024**2,seconds=time.time()-start)),flush=True)
 @torch.no_grad()
 def batch(self,ids,mode,canonical=False):
  if isinstance(ids,np.ndarray):ids=np.ascontiguousarray(ids)
  ids=torch.as_tensor(ids,device='cuda',dtype=torch.long);n=len(ids);x=self.x[ids]
  if mode=='none':
   if canonical:
    xx=torch.zeros((256,206),device='cuda');xx[:n]=x;x=xx
   return x,torch.zeros((len(x),64,38),device='cuda',dtype=torch.float16)
  if canonical:
   assert n<=256
   xx=torch.zeros((256,206),device='cuda');xx[:n]=x
   xy=torch.zeros((256,4),device='cuda');hour=torch.zeros(256,device='cuda');weekend=torch.zeros(256,device='cuda',dtype=torch.bool)
   p=torch.full((256,64),-1,device='cuda',dtype=self.parent.dtype)
   first=torch.zeros((256,64),device='cuda',dtype=self.first.dtype);last=torch.zeros_like(first)
   xy[:n]=self.xy[ids];hour[:n]=self.hour[ids];weekend[:n]=self.weekend[ids]
   p[:n]=self.parent[ids];first[:n]=self.first[ids];last[:n]=self.last[ids]
   x=xx
  else:
   xy=self.xy[ids];hour=self.hour[ids];weekend=self.weekend[ids]
   p=self.parent[ids];first=self.first[ids];last=self.last[ids]
  raw=self.bank.features(xy,hour,weekend,p,first,last,geometry_proxy=mode=='geometry')
  raw=add_public_compatibility(raw,x,self.shape_indices,self.available_indices)
  return x,raw
 @torch.no_grad()
 def predict(self,model,mode,ids=None):
  from cuantiles_mezcla_balanceada import stable_ensemble_quantiles
  model.eval();ids=np.arange(self.nfit,self.n) if ids is None else np.asarray(ids);parts=[]
  for a in range(0,len(ids),256):
   block=ids[a:a+256];xx,rr=self.batch(block,mode,canonical=True)
   parts.append(stable_ensemble_quantiles(model(xx,rr))[:len(block)].cpu())
  return torch.cat(parts).numpy()
