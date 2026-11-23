from base_replanteo import *
p=OUT/'inferencia_atencion_subtrayectos.py';s=p.read_text()
s=s.replace("def __init__(self,run_id='atencion_subtrayectos_s0'):","def __init__(self,run_id='atencion_subtrayectos_s0',allow_pending=False):")
s=s.replace("  self.run=run;self.record=r=read(run/'resultado.json');assert r['complete'] and r['qualified']",
 """  self.run=run
  if allow_pending:
   assert not (run/'resultado.json').exists()
   config=read(run/'protocolo.json');history=read(run/'historia.json')
   assert len(history)==6 and all(v['primary_train_n']==1023485 for v in history)
   path=run/'epoca_6.pt';checkpoint=torch.load(path,map_location='cpu',weights_only=False)
   recovery=torch.load(run/'ultima_reanudacion.pt',map_location='cpu',weights_only=False)
   assert checkpoint['epoch']==recovery['epoch']==6 and checkpoint['config']==config==recovery['config']
   for key in checkpoint['branch_state']:assert torch.equal(checkpoint['branch_state'][key],recovery['branch_state'][key])
   sources={p.name:sha(p) for p in (run/'fuentes').glob('*.py')}
   for name,digest in sources.items():assert sha(OUT/name)==digest
   r=dict(complete=False,qualified=False,config=config,model_file=str(path),model_sha256=sha(path),sources=sources)
   write(run/'recuperacion_evaluacion_protocolo.json',dict(reload_epoch=6,model_sha256=r['model_sha256'],saved_branch_matches_final_optimizer_checkpoint=True,
    training_not_repeated=True,reason='FinalB2048distribution calculation exceeded6GiBcap withallTrain tensors resident; reloadonlyValidationdata.',
    same_architecture_weights_batch_and_query_information=True,Test='sealed'))
  else:
   r=read(run/'resultado.json');assert r['complete'] and r['qualified']
  self.record=r""")
p.write_text(s)
prefix=(OUT/'reproducir_atencion_subtrayectos.py').read_text().split("p=np.concatenate(parts);saved=")[0]
prefix=prefix.replace("expert=FrozenSubtrajectoryExpert()","expert=FrozenSubtrajectoryExpert(allow_pending=True)")
ending="""
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
"""
p=OUT/'recuperar_evaluacion_atencion_subtrayectos.py';assert not p.exists();p.write_text(prefix+ending)
import ast
for name in ['inferencia_atencion_subtrayectos.py','recuperar_evaluacion_atencion_subtrayectos.py']:ast.parse((OUT/name).read_text())
print('Isolatedevaluation recovery prepared; no training rerun.')
