"""Causal Train-pilot inputs; no persistent token cache or heldtarget feature."""
from base_replanteo import *
import torch,gc
from inferencia import feature_input
from tokens_subtrayectos import query_parts_sub
from tokens_transferencia_arco import ArcTransferBank,add_public_compatibility,compatibility_indices
def build_public206(items,cfg):
 from consulta_od import od_records,build_od_features
 from consulta_historicos_cruzados import prepare,history_features,STATIC_NAMES
 config=dict(cfg);config.update(projected_routes=False,mapmatched=False,learned_route=False,road_attributes=False)
 data=od_records(items);base,names=build_od_features(data,config);_,static=history_features(prepare(items),5)
 return np.column_stack([base,static]).astype(np.float32),names+STATIC_NAMES
def tensor_tokens(bank,queries,x,parent,first,last,shape_indices,available_indices,geometry_proxy=False):
 n=len(x);assert n<=256
 xy=torch.zeros((256,4),device=bank.xy.device);hour=torch.zeros(256,device=bank.xy.device);weekend=torch.zeros(256,device=bank.xy.device,dtype=torch.bool)
 parts=query_parts_sub(queries)
 xy[:n]=torch.tensor(parts['xy_m'],device=xy.device);hour[:n]=torch.tensor(parts['hour'],device=xy.device);weekend[:n]=torch.tensor(parts['weekend'],device=xy.device)
 p=np.full((256,64),-1,np.int32);i=np.zeros((256,64),np.uint8);j=np.zeros_like(i)
 p[:n]=parent;i[:n]=first;j[:n]=last
 xx=torch.zeros((256,206),device=xy.device);xx[:n]=torch.as_tensor(x,device=xy.device)
 raw=bank.features(xy,hour,weekend,p,i,j,geometry_proxy=geometry_proxy)
 return add_public_compatibility(raw,xx,shape_indices,available_indices)[:n]
def prepare_inputs(run):
 start=time.time();source=OUT/'oraculo_tiempo_calle_train_s0'
 assert sha(source/'auxiliares_Train.npz')==read(source/'resultado.json')['auxiliary_sha256']
 with np.load(source/'auxiliares_Train.npz') as z:q={k:z[k] for k in ['row','train_index','timestamp','true']}
 cutoff=int(q['timestamp'][40000:].min());assert np.max(q['timestamp'][:40000]+q['true'][:40000])<cutoff
 pilot=OUT/'piloto_cdf_ordenada_Train_s0';rec=read(pilot/'resultado.json')
 assert sha(pilot/'indices_fuentes.npz')==rec['sourceindices_sha256']
 assert read(pilot/'verificacion_fuentes_features.json')['passed']
 with np.load(pilot/'indices_fuentes.npz') as z:
  for k in ['row','timestamp']:assert np.array_equal(z[k],q[k])
  refs={k:z[k] for k in ['parent','first','last']}
 d,X,names=feature_input('train',read(BEST/'resultado.json'));keep=np.array([not name.startswith('mm_') for name in names])
 names=[name for name,k in zip(names,keep) if k];assert len(names)==206
 X=np.ascontiguousarray(X[q['train_index']][:,keep]);dd={k:d[k][q['train_index']] for k in ['o_lon','o_lat','d_lon','d_lat','timestamp']}
 assert np.array_equal(d['row'][q['train_index']],q['row'])
 assert np.array_equal(np.load(DATA/'train_y.npy')[q['train_index']],q['true'])
 sourcefile=OUT/'memoria_puntos_Train_s0/puntos_Train.npz';source_meta=read(sourcefile.parent/'resultado.json')
 assert sha(sourcefile)==source_meta['outputs_sha256']
 with np.load(sourcefile) as z:
  stamps=z['timestamp'];times=z['time_s'];sizes=z['parent_gps_n'];folds=z['parent_fold'];offset=z['offsets']
  eligible=np.zeros(len(stamps),bool);eligible[z['eligible']]=True
  inside=np.unpackbits(z['indexable_packed'],count=int(offset[-1])).astype(bool)
 p=refs['parent'];valid=p>=0;safe=np.maximum(p,0);first=refs['first'].astype(np.int16);last=refs['last'].astype(np.int16)
 assert eligible[safe[valid]].all()
 assert np.all(stamps[safe[valid]]+times[safe[valid]]<cutoff)
 assert np.all((last-first)[valid]>=4) and np.all((last-first)[valid]<=119)
 assert np.all(last[valid]<sizes[safe[valid]]) and np.all(first[valid]>=0)
 assert inside[(offset[safe]+first)[valid]].all() and inside[(offset[safe]+last)[valid]].all()
 queryfold=((dd['timestamp'][:40000]//604800)%5)[:,None];fitvalid=valid[:40000]
 assert np.all(folds[safe[:40000]][fitvalid]!=np.broadcast_to(queryfold,fitvalid.shape)[fitvalid])
 assert not np.any((p==q['train_index'][:,None])&valid)
 del inside,offset,first,last,safe,eligible;gc.collect()
 shape_idx,available_idx=compatibility_indices(names)
 bank=ArcTransferBank('cuda')
 gx=torch.tensor(X,device='cuda');observed=torch.empty((50000,64,38),device='cuda',dtype=torch.float16);geometry=torch.empty_like(observed)
 stats={key:dict(n=0,total=np.zeros(38),square=np.zeros(38)) for key in ['observed','geometry']}
 for a in range(0,50000,256):
  b=min(a+256,50000);queries={k:v[a:b] for k,v in dd.items()};rr={k:v[a:b] for k,v in refs.items()}
  actual=tensor_tokens(bank,queries,X[a:b],**rr,shape_indices=shape_idx,available_indices=available_idx)
  control=tensor_tokens(bank,queries,X[a:b],**rr,shape_indices=shape_idx,available_indices=available_idx,geometry_proxy=True)
  untouched=[j for j in range(38) if j not in [16,32]]
  assert torch.equal(actual[...,untouched],control[...,untouched])
  assert np.array_equal((actual[...,27]>.5).cpu().numpy(),valid[a:b])
  observed[a:b]=actual;geometry[a:b]=control
  fit_n=max(0,min(b,40000)-a)
  if fit_n:
   for key,raw in [('observed',actual),('geometry',control)]:
    values=raw[:fit_n].float().cpu().numpy().reshape(-1,38);values=values[values[:,27]>.5].astype(float)
    stats[key]['n']+=len(values);stats[key]['total']+=values.sum(0);stats[key]['square']+=(values**2).sum(0)
  if a//256%40==0:print(json.dumps(dict(stage='arc_token_inputs',n=b,seconds=time.time()-start)),flush=True)
 norm={}
 for key,value in stats.items():
  mean=value['total']/value['n'];std=np.sqrt(np.maximum(0,value['square']/value['n']-mean**2));std[std<1e-4]=1
  norm[key]=dict(mean=mean.astype(np.float32),std=std.astype(np.float32),fit_source_tokens=value['n'])
 assert norm['observed']['fit_source_tokens']==int(valid[:40000].sum())
 mean=X[:40000].mean(0,dtype=np.float64);std=X[:40000].std(0,dtype=np.float64);std[std<1e-4]=1
 anchor_names=['network_fastest_time_600s','projected_length_freeflow_600s','projected_free_freeflow_600s','od_km']
 anchor_indices=np.repeat([names.index(n) for n in anchor_names],2);anchor_scale=np.repeat([600.,600.,600.,1000.],2)
 coefficients=[]
 for index,scale in zip(anchor_indices,anchor_scale):
  value=np.log(np.maximum(.1,X[:40000,index].astype(float)*scale))
  coef=np.linalg.lstsq(np.column_stack([np.ones(40000),value]),np.log(q['true'][:40000]),rcond=None)[0];coefficients.append(coef)
 ctor=dict(xmean=mean.astype(np.float32),xstd=std.astype(np.float32),anchor_indices=anchor_indices,
  anchor_scale=anchor_scale.astype(np.float32),anchor_coef=np.asarray(coefficients,np.float32))
 torch.save(dict(common=ctor,source_normalization=norm,features=names,shape_indices=shape_idx,available_indices=available_idx,source_cutoff=cutoff),run/'constructor.pt')
 del bank;torch.cuda.empty_cache()
 write(run/'verificacion_datos.json',dict(passed=True,fit_n=40000,held_n=10000,fit_source_tokens=int(valid[:40000].sum()),
  no_self_source=True,all_source_arrivals_before_heldcutoff=True,all_fit_parent_weekfolds_excluded=True,sourceendpoint_mask_and_span_preserved=True,
  raw_feature_change_control_only_columns=[16,32],normalizers_and_time_anchors_fit40000only=True,
  constructor_sha256=sha(run/'constructor.pt'),sourceindices_sha256=rec['sourceindices_sha256'],source_pointbank_sha256=source_meta['outputs_sha256'],
  prefix_sha256=read(OUT/'distancia_arco_puntos_Train_s0/resultado.json')['outputs_sha256'],
  held_targets_never_input=True,persistent_raw_token_tensor=False,GPU_resident_input_MiB=torch.cuda.memory_allocated()/1024**2,seconds=time.time()-start))
 return dict(X=gx,observed=observed,geometry=geometry,y=np.asarray(q['true'],np.float32),q=q,queries=dd,refs=refs,constructor=ctor,norm=norm,names=names)
