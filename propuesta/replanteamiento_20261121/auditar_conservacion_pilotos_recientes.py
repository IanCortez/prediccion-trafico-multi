"""Recompute completed recent Train metrics; retains every result regardless131.06."""
from base_replanteo import *
access=deny_test();records=[]
def verify(run,archive,metrics,expected_sha=None,variant='original'):
 file=run/archive;assert file.resolve().is_relative_to(ROOT.resolve()) and file.exists()
 digest=sha(file)
 if expected_sha is not None:assert digest==expected_sha,(run.name,archive)
 z=dict(np.load(file));assert z['true'].shape==(10000,) and len(z['row'])==10000
 checked={}
 for mode,expected in metrics.items():
  key=mode if mode in z else 'pred'
  actual=metric(z['true'],z[key]);assert actual==expected,(run.name,mode,actual,expected)
  checked[mode]=dict(metrics=actual,above131_06=actual['mae_s']>131.06)
 records.append(dict(id=run.name,variant=variant,predictions=str(file.relative_to(ROOT)),sha256=digest,
  bytes=file.stat().st_size,statistics=checked,all_metrics_recomputed_exact=True,
  scope='Reused heldTrain; not primaryValidation or Test',retained=True))
for name in ['piloto_transferencia_intervalos_Train_s0','piloto_cdf_ordenada_Train_s0','mezcla_individual_mediana_Train_s0','puente_temporal_2d_Train_s0']:
 run=OUT/name;r=read(run/'resultado.json');assert r['complete']
 verify(run,'predicciones_held_Train.npz',r['metrics'],r['predictions_sha256'])
 plans=[p for p in [run/'protocolo.json',run/'plan.json'] if p.exists()];assert len(plans)>=1
 records[-1]['configuration_files']={p.name:sha(p) for p in plans}
 if name=='mezcla_individual_mediana_Train_s0':
  replay=read(run/'reproduccion_independiente.json');assert replay['passed']
  verify(run,'predicciones_balanceadas_held_Train.npz',replay['stable_metrics'],variant='stable_sameparameters')
run=OUT/'mezcla_individual_escala_Train_s0'
if (run/'resultado.json').exists():
 r=read(run/'resultado.json');assert r['complete'];verify(run,'predicciones_held_Train.npz',r['metrics'],r['predictions_sha256'])
 current='complete_all_three_arms'
else:
 current='in_progress; only completed arms audited'
 for mode in ['none','geometry','observed']:
  p=run/mode/'resultado.json'
  if not p.exists():continue
  r=read(p);assert r['complete'];verify(p.parent,'predicciones_held_Train.npz',{mode:r['metrics']},r['predictions_sha256'])
run=OUT/'control_familia_puente_2d_Train_s0'
r=read(run/'resultado.json');assert r['complete']
verify(run,'predicciones_held_Train.npz',{'matched_lognormal_control':r['metrics']},r['predictions_sha256'])
records[-1]['configuration_files']={'protocolo.json':sha(run/'protocolo.json')}
run=OUT/'red_cinetica_Train_s0'
if (run/'resultado.json').exists():
 r=read(run/'resultado.json');assert r['complete']
 verify(run,'predicciones_held_Train.npz',r['metrics'],r['predictions_sha256'])
 records[-1]['configuration_files']={'protocolo.json':sha(run/'protocolo.json')}
run=OUT/'red_cinetica_Train_s0'
if (run/'diagnostico_sin_compuerta.json').exists():
 r=read(run/'diagnostico_sin_compuerta.json');assert r['complete']
 verify(run,'predicciones_red_sin_compuerta_held_Train.npz',r['metrics'],r['predictions_sha256'],variant='postfixed_rawgraph_diagnostic')
report=dict(passed=True,all_registered_completed_results_retained=True,MAE_threshold_excludes_no_results=True,
 completed_prediction_archives=len(records),worse_than131_06_outputs=sum(s['above131_06'] for r in records for s in r['statistics'].values()),
 records=records,current_family=current,weights='Separate retention policy; no weight or cache deletion by this audit.',
 existing_primary_index='registro_consolidado_auditado.json',Test='SEALED',Validation=False,
 actual_data_files_accessed=sorted(access),source_sha256=sha(__file__))
write(OUT/'conservacion_pilotos_recientes.json',report)
print(json.dumps({key:report[key] for key in ['passed','completed_prediction_archives','worse_than131_06_outputs','current_family']}))
