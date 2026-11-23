"""Independent full-checkpoint reload and full-primary/raw5 proof for a qualified joint model."""
from base_replanteo import *
import torch,argparse,shutil
from inferencia import feature_input
from inferencia_subtrayectos_conjuntos import FrozenJointSubtrajectory
p=argparse.ArgumentParser();p.add_argument('--run',default='subtrayectos_conjuntos_s0');a=p.parse_args()
access=deny_test();start=time.time();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction(.5)
expert=FrozenJointSubtrajectory(a.run);run=expert.run;r=expert.record
d,V,names=feature_input('validation',read(BEST/'resultado.json'));assert names==expert.features
C=np.load(OUT/'banco_proyeccion_incierta_s0/validation_banco.npz')['bank']
H=np.load(OUT/'datos_prototipos_historicos_s0/validation_historicos.npz')['history']
indexfile=OUT/'indices_subtrayectos_s0/validation_indices.npz';meta=read(OUT/'indices_subtrayectos_s0/resultado.json')
assert sha(indexfile)==meta['outputs']['validation']['sha256']
S=dict(np.load(indexfile));sq=expert.tokens.query_parts_sub(d)
for key in ['row','timestamp']:assert np.array_equal(d[key],S[key])
parts=[]
with torch.no_grad():
 for first in range(0,len(V),2048):
  n=min(2048,len(V)-first);index=np.arange(first,first+n)
  if n<2048:index=np.r_[index,np.repeat(index[-1],2048-n)]
  sub=expert.library.features(*[torch.as_tensor(sq[key][index],device='cuda') for key in ['xy_m','hour','weekend']],
   S['parent'][index],S['first'][index],S['last'][index])
  sub=expert.adjust_tokens(sub)
  out=expert.model(*[torch.as_tensor(v[index],device='cuda') for v in [V,C,H]],sub)
  parts.append(expert.model.duration.quantiles(out[0])[:n].cpu().numpy())
q=np.concatenate(parts);saved=dict(np.load(run/'validacion.npz'))
for key in ['row','timestamp']:assert np.array_equal(d[key],saved[key])
truth=np.load(DATA/'validation_y.npy');assert np.array_equal(truth,saved['true'])
assert np.array_equal(q,saved['quantiles']) and np.array_equal(q[:,1],saved['pred'])
assert metric(truth,q[:,1])==r['validation']
ii=np.sort(np.random.default_rng(20261122).choice(len(q),65,replace=False));items=read(run/'consultas_raw_sin_etiquetas.json')
raw=expert.raw(items);rev=expert.raw(items[::-1]);single=expert.raw(items[:1])
assert np.array_equal(raw,q[ii]) and np.array_equal(raw,rev[::-1]) and np.array_equal(raw[:1],single)
artifact=run/'reproduccion_fuentes';artifact.mkdir(exist_ok=False)
for name in ['inferencia_subtrayectos_conjuntos.py','reproducir_subtrayectos_conjuntos.py']:shutil.copy2(OUT/name,artifact/name)
result=dict(passed=True,reloaded_model_sha256=r['model_sha256'],all127941quantiles_exact=True,raw65_reverse_singleton_exact=True,
 validation=r['validation'],sources={p.name:sha(p) for p in artifact.glob('*.py')},Test_access=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(run/'reproduccion_independiente.json',result);print(json.dumps(result),flush=True)
