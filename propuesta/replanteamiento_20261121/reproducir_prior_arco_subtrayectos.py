"""Reload saved qualified branch and verify every full-primary quantile plus raw-fivefield invariance."""
from base_replanteo import *
import torch
from inferencia import feature_input
from inferencia_prior_arco_subtrayectos import FrozenArcPriorExpert
access=deny_test();start=time.time();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction(.5)
expert=FrozenArcPriorExpert();run=expert.run;r=expert.record
d,V,names=feature_input('validation',read(BEST/'resultado.json'));assert names==expert.features
C=np.load(OUT/'banco_proyeccion_incierta_s0/validation_banco.npz')['bank']
H=np.load(OUT/'datos_prototipos_historicos_s0/validation_historicos.npz')['history']
Q=dict(np.load(OUT/'datos_historia_individual_s0/validation_indices.npz'))
S=dict(np.load(OUT/'indices_subtrayectos_s0/validation_indices.npz'));sq=expert.tokens.query_parts_sub(d)
for key in ['row','timestamp']:
 assert np.array_equal(d[key],Q[key]) and np.array_equal(d[key],S[key])
parts=[]
with torch.no_grad():
 for first in range(0,len(V),2048):
  n=min(2048,len(V)-first);index=np.arange(first,first+n)
  if n<2048:index=np.r_[index,np.repeat(index[-1],2048-n)]
  q={k:torch.as_tensor(Q[k][index],device='cuda') for k in ['xy_km','hour','weekend','indices']}
  old=expert.oldlibrary.features(q['xy_km'],q['hour'],q['weekend'],q['indices'])
  sub=expert.library.features(*[torch.as_tensor(sq[k][index],device='cuda') for k in ['xy_m','hour','weekend']],
   S['parent'][index],S['first'][index],S['last'][index])
  geometry=expert.arc_library.features(*[torch.as_tensor(sq[k][index],device='cuda') for k in ['xy_m','hour','weekend']],
   S['parent'][index],S['first'][index],S['last'][index])[:,:,:14]
  out=expert.model(*[torch.as_tensor(v[index],device='cuda') for v in [V,C,H]],old,sub,geometry)
  parts.append(expert.model.duration.quantiles(out[0])[:n].cpu().numpy())
p=np.concatenate(parts);saved=dict(np.load(run/'validacion.npz'))
for key in ['row','timestamp']:assert np.array_equal(saved[key],d[key])
assert np.array_equal(saved['quantiles'],p) and np.array_equal(saved['pred'],p[:,1])
assert metric(saved['true'],p[:,1])==r['validation']
items=read(run/'consultas_raw_sin_etiquetas.json');ii=np.sort(np.random.default_rng(20261122).choice(len(p),65,replace=False))
raw=expert.raw(items);rev=expert.raw(items[::-1]);single=expert.raw(items[:1])
assert np.array_equal(raw,p[ii]) and np.array_equal(raw,rev[::-1]) and np.array_equal(raw[:1],single)
result=dict(passed=True,reloaded_model_sha256=r['model_sha256'],all127941quantiles_exact=True,
 raw65_reverse_singleton_exact=True,validation=r['validation'],source_sha256=sha(__file__),
 inference_source_sha256=sha(OUT/'inferencia_prior_arco_subtrayectos.py'),Test_access=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(run/'reproduccion_independiente.json',result);print(json.dumps(result),flush=True)
