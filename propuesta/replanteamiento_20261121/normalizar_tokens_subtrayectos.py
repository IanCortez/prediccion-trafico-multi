"""Train-only source normalization and calibrated duration anchor; fixed selection."""
from base_replanteo import *
import torch,shutil
from tokens_subtrayectos import SubtripTokenBank,query_parts_sub
access=deny_test();start=time.time();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.5)
folder=OUT/'indices_subtrayectos_s0';meta=read(folder/'resultado.json');assert meta['complete']
z=dict(np.load(folder/'train_indices.npz'));assert sha(folder/'train_indices.npz')==meta['outputs']['train']['sha256']
d=dict(np.load(DATA/'train_x.npz'));y=np.load(DATA/'train_y.npy')
for key in ['row','timestamp']:assert np.array_equal(z[key],d[key])
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index']
ii=np.sort(np.random.default_rng(5833).choice(eligible,32768,replace=False))
part=query_parts_sub(d);bank=SubtripTokenBank();parts={k:torch.as_tensor(v,device='cuda') for k,v in part.items()}
sums=np.zeros(34);squares=sums.copy();count=0;anchor_x=[];anchor_y=[];raw17=None
for first in range(0,len(ii),512):
 index=ii[first:first+512];raw=bank.features(parts['xy_m'][index],parts['hour'][index],parts['weekend'][index],
  z['parent'][index],z['first'][index],z['last'][index])
 v=raw.float().cpu().numpy();valid=v[:,:,27]>.5
 a=v[valid].astype(np.float64);assert np.isfinite(a).all()
 sums+=a.sum(0);squares+=(a*a).sum(0);count+=len(a)
 logits=np.where(valid,v[:,:,26],-1e30).astype(np.float64)
 weights=np.exp(logits-logits.max(1,keepdims=True))*valid;weights/=np.maximum(weights.sum(1,keepdims=True),1e-30)
 have=valid.any(1);anchor_x.extend((v[:,:,16]*weights).sum(1)[have]);anchor_y.extend(np.log(y[index][have]))
 if first==0:
  raw17=raw[:17].clone()
  reverse=bank.features(parts['xy_m'][index[:17][::-1].copy()],parts['hour'][index[:17][::-1].copy()],parts['weekend'][index[:17][::-1].copy()],
   z['parent'][index[:17][::-1]],z['first'][index[:17][::-1]],z['last'][index[:17][::-1]])
  single=bank.features(parts['xy_m'][index[:1]],parts['hour'][index[:1]],parts['weekend'][index[:1]],
   z['parent'][index[:1]],z['first'][index[:1]],z['last'][index[:1]])
  assert torch.equal(raw17,reverse.flip(0)) and torch.equal(raw17[:1],single)
mean=sums/count;std=np.maximum(np.sqrt(np.maximum(squares/count-mean*mean,0)),.03)
A=np.column_stack([np.ones(len(anchor_x)),anchor_x]);b=np.array(anchor_y);beta=np.linalg.lstsq(A,b,rcond=None)[0]
for iteration in range(15):
 residual=b-A@beta;scale=max(.05,1.4826*np.median(np.abs(residual-np.median(residual))))
 weight=np.minimum(1.,1.345*scale/np.maximum(np.abs(residual),1e-8))
 beta=np.linalg.solve(A.T@(weight[:,None]*A)+1e-6*np.eye(2),A.T@(weight*b))
beta[1]=np.clip(beta[1],.25,1.5);beta[0]=np.median(b-beta[1]*A[:,1])
target=folder/'constructor_tokens.npz';assert not target.exists()
np.savez_compressed(target,mean=mean.astype(np.float32),std=std.astype(np.float32),time_anchor=beta.astype(np.float32),normalization_query_indices=ii)
result=dict(passed=True,normalization_n=32768,valid_source_tokens=count,anchor_n=len(b),time_anchor=beta,
 source_tokens_finite=True,source_reverse_singleton_exact=True,all_source_context_columns_Trainonly=True,
 constructor_sha256=sha(target),sources={name:sha(OUT/name) for name in ['tokens_subtrayectos.py','normalizar_tokens_subtrayectos.py']},
 no_Validationlabels=True,Test_access=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(folder/'normalizacion_resultado.json',result);print(json.dumps(read(folder/'normalizacion_resultado.json')),flush=True)
