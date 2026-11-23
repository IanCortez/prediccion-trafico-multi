"""Reload saved qualified branch and verify every full-primary quantile plus raw-fivefield invariance."""
from base_replanteo import *
import torch
from inferencia import feature_input
from inferencia_atencion_subtrayectos import FrozenSubtrajectoryExpert
access=deny_test();start=time.time();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction(.5)
expert=FrozenSubtrajectoryExpert(allow_pending=True);run=expert.run;r=expert.record
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
  out=expert.model(*[torch.as_tensor(v[index],device='cuda') for v in [V,C,H]],old,sub)
  parts.append(expert.model.duration.quantiles(out[0])[:n].cpu().numpy())

p=np.concatenate(parts)
truth=np.load(DATA/'validation_y.npy');reference=read(OUT/r['config']['parent_id']/'resultado.json')
saved=dict(np.load(OUT/r['config']['parent_id']/'validacion.npz'))
for key in ['row','timestamp']:assert np.array_equal(d[key],saved[key])
assert np.array_equal(truth,saved['true']) and len(truth)==127941
ii=np.sort(np.random.default_rng(20261122).choice(len(p),65,replace=False))
fields=['o_lon','o_lat','d_lon','d_lat','timestamp']
items=[{key:int(d[key][i]) if key=='timestamp' else float(d[key][i]) for key in fields} for i in ii]
raw=expert.raw(items);rev=expert.raw(items[::-1]);single=expert.raw(items[:1])
assert np.array_equal(raw,p[ii]) and np.array_equal(raw,rev[::-1]) and np.array_equal(raw[:1],single)
m=metric(truth,p[:,1]);comparison=paired(truth,p[:,1],saved['pred'],d['timestamp'])
gate=comparison['delta_mae_s']<=-.5 and comparison['ci95_s'][1]<0
qualified=m['mae_s']<131.06 or m['mae_s']<reference['validation']['mae_s']
np.savez_compressed(run/'validacion.npz',row=d['row'],timestamp=d['timestamp'],true=truth,pred=p[:,1],quantiles=p,reference=saved['pred'])
result=dict(complete=True,id=run.name,config=r['config'],validation=m,paired_vs_parent=comparison,meaningful_gate_passed=bool(gate),
 qualified=bool(qualified),model_file=r['model_file'] if qualified else None,model_sha256=r['model_sha256'],fixed_epoch=6,
 parent_unchanged=sha(reference['model_file'])==reference['model_sha256'],parent_source=reference['id'],
 raw65_cached_sources_tokens_full_quantiles_exact=True,raw_reverse_singleton_exact=True,
 all_original_primary_queries_and_T=True,Train_parent_source_fold_excluded=True,
 normalization_sha256=r['config']['normalization_sha256'],source_manifest_sha256=r['config']['source_manifest_sha256'],
 sources=r['sources'],recovery_evaluation_source_sha256=sha(__file__),predictions_sha256=sha(run/'validacion.npz'),
 uncertainty=dict(coverage80=float(np.mean((truth>=p[:,0])&(truth<=p[:,2]))),width80_s=float((p[:,2]-p[:,0]).mean())),
 training_epochs_all_complete=6,training_not_repeated=True,isolated_evaluation_recovery=True,Test_access=False,
 actual_data_files_accessed=sorted(access),evaluation_seconds=time.time()-start)
write(run/'resultado.json',result);write(run/'consultas_raw_sin_etiquetas.json',items)
write(run/'reproduccion_independiente.json',dict(passed=True,reloaded_model_sha256=r['model_sha256'],all127941quantiles_fromfinal_savedcheckpoint=True,
 raw65_reverse_singleton_exact=True,validation=m,originalTraining_rerun=False,Test_access=False))
removed=[]
path=Path(r['model_file']).resolve()
if not qualified:
 assert path.is_relative_to(run.resolve()) and sha(path)==r['model_sha256'];path.unlink();removed.append(dict(file=path.name,sha256=r['model_sha256']))
path=(run/'ultima_reanudacion.pt').resolve();assert path.is_relative_to(run.resolve());digest=sha(path);path.unlink();removed.append(dict(file=path.name,sha256=digest))
write(run/'limpieza.json',dict(removed=removed,all_completed_metrics_predictions_sources_retained=True))
print(json.dumps(read(run/'resultado.json')),flush=True)
