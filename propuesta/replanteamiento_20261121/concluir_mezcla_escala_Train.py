"""Close only after independent replay and exact audit; retain every outcome/prediction."""
from base_replanteo import *
run=OUT/'mezcla_individual_escala_Train_s0'
assert not (run/'conclusion.json').exists(),'One-shot scientific closure; do not repeat retirement.'
r=read(run/'resultado.json');a=read(run/'auditoria_resultado.json');replay=read(run/'reproduccion_independiente.json')
assert r['complete'] and a['complete'] and replay['passed']
assert a['practical_gate_passed']==r['passed']
assert sha(run/'predicciones_held_Train.npz')==r['predictions_sha256']
assert sha(OUT/'atencion_subtrayectos_s0/epoca_6.pt')=='07db7a1d4f77647de15a1db6c8be1b68e26bdd4edf16a43aab2b0d2f905c39dd'
retention=dict(policy='Keep every completedmetric/prediction/config/source, includingMAE>131.06. PilotheldTrain is not fullprimaryValidation for weightqualification.',
 retired=[],preserved_weights=[],retired_bytes=0)
for item in r['checkpoints']:
 file=Path(item['file']).resolve()
 assert file.is_relative_to(run.resolve()) and file.name=='epoca_4.pt' and sha(file)==item['sha256']
 record=dict(arm=item['arm'],file=str(file.relative_to(ROOT)),sha256=item['sha256'],bytes=file.stat().st_size)
 if r['passed']:
  retention['preserved_weights'].append(record)
 else:
  # Journal exact retirement BEFORE unlink; no glob, recursive operation or unrelatedpath.
  retention['retired'].append(record);write(run/'retencion_resultado.json',retention)
  file.unlink();retention['retired_bytes']+=record['bytes']
if not r['passed']:
 file=(OUT/'indices_escala_Train_s0/indices_fuentes.npz').resolve()
 assert file.is_relative_to(OUT.resolve()) and file.name=='indices_fuentes.npz'
 assert sha(file)==r['sourceindices_sha256']
 # This is an experiment-specific construction cache, absent from every protected modelregistry.
 for registry in OUT.glob('*protegidos.json'):
  assert 'indices_escala_Train_s0' not in registry.read_text(encoding='utf-8-sig')
 meta_file=file.parent/'metadatos_Train.npz'
 assert not meta_file.exists()
 with np.load(file) as z:np.savez_compressed(meta_file,row=z['row'],timestamp=z['timestamp'],train_index=z['train_index'],fit_n=r['fit_n'])
 retention['retained_fit_metadata']=dict(file=str(meta_file.relative_to(ROOT)),sha256=sha(meta_file),bytes=meta_file.stat().st_size)
 cache=dict(file=str(file.relative_to(ROOT)),sha256=r['sourceindices_sha256'],bytes=file.stat().st_size,
  kind='Discarded fullpilot compactindexcache; rebuild via preparar_indices_escala_Train.py before repeating its replay.',
  source_and_manifest_retained=True)
 retention['retired'].append(cache);write(run/'retencion_resultado.json',retention)
 file.unlink();retention['retired_bytes']+=cache['bytes']
write(run/'retencion_resultado.json',retention)
r.update(independent_replay_pending=False,independent_replay_sha256=sha(run/'reproduccion_independiente.json'),
 retention_record='retencion_resultado.json',sourcecache_retired=not r['passed'])
write(run/'resultado.json',r)
a['retention_decision_pending']=False;a['retention_record']='retencion_resultado.json';write(run/'auditoria_resultado.json',a)
c=dict(complete=True,practical_gate=bool(r['passed']),metrics=r['metrics'],comparisons=r['comparisons'],
 change_vs_smallfit_held_MAE_s=a['change_vs_smallfit_held_MAE_s'],
 independent_full10000_raw65_reverse_singleton_replay_passed=True,
 original_rows_labels_and_query_information_preserved=True,
 critical_interpretation=[
 'The full951718fit comparison improves the generalizationregime relative to the small40k pilot; changing samplecomposition and stabletrainingnumerics together does not identify a pure causal samplesize effect.',
 'Observedsource time must beat both freshneuralbaseline and geometryproxy by the original declaredgates, not merely beat the earlier weaksmallpilot.',
 'All heldTrain results are exploratory, repeatedlyreused and on one initialization. They do not replace fullValidation or Testconfirmation.',
 'A negative gate does not prove an irreducibleBayeserror floor or that73.5 is universallyimpossible.',
 'A positive gate only qualifies a prospectivelyfrozen fullprimaryexperiment, not a73.5claim.'
 ],
 next='Preregister fullunchangedprimary experiment: fit exactly1023485eligibleprimaryTrain and evaluate127941Validation once at fixedfinalstate, with matchedcontrols, after fullsourceconstructorproof.' if r['passed'] else
 'Do not sweep epochs, sourcecount, solver or thresholds. Review a structurallydistinct, physicallyconsistent alternative before its prospectivelycontrolled Trainexperiment. Basic position-only Markovwalk is notready after kineticdiagnostic.',
 all_metrics_predictions_configs_frozen_sources_retained=True,
 weights_and_cache_retention=retention,goal='ACTIVE',primary_best_unchanged_s=129.99003446901307,
 confirmation_goal_s=73.5,confirmation_goal_achieved=False,Test='SEALED',Validation=False)
write(run/'conclusion.json',c)
state=read(OUT/'CONTINUIDAD_OPERATIVA.json')
state['current_trial']=dict(experiment=run.name,status='CLOSED_AFTER_INDEPENDENT_REPLAY',practical_gate=bool(r['passed']),Test='SEALED')
state['last_closed_timepilot_id']=run.name;state['last_closed_timepilot']=c
state['next_after_result']=[
 'Originalprimarybest129.990034 unchanged, finalconfirmation73.5unmet. Testsealed.',
 c['next'],
 'Retain every completedmetric/prediction, includingabove131.06; preserve originalbest lineage, qualifiedfullprimarymodels and10TransferTraj archives.',
 'New mathematicalwalkpartition instrumentpassed, but its position-only physicalrepresentation is notready: strongheadingmemory, varying roadedgelength and noisyGPS require a correct clock/emissionmodel first.'
]
state['current_alternative_review']='evaluacion_previa_marginalizacion_caminos.json'
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
active=read(OUT/'arquitectura_activa.json')
active['current_trial']=dict(experiment=run.name,status='CLOSED',practical_gate=bool(r['passed']),conclusion=str((run/'conclusion.json').relative_to(OUT)))
active['next']=c['next'];active['Test']='SEALED';write(OUT/'arquitectura_activa.json',active)
plan=read(OUT/'plan_escala_datos_Train.json')
plan['status']='CLOSED_AFTER_INDEPENDENT_REPLAY';plan['practical_gate_passed']=bool(r['passed'])
plan['outcome_manifest']=str((run/'conclusion.json').relative_to(OUT));write(OUT/'plan_escala_datos_Train.json',plan)
marker='<!-- resultado_mezcla_escala_Train_s0 -->'
text=marker+'\n## Mezcla individual: comparacion con todo Train previo al corte\n\n'
text+='Con 951.718 viajes de ajuste y las mismas 10.000 consultas cronologicas de reserva, tras cuatro epocas completas fijas, los MAE fueron: '
text+=', '.join(mode+' '+format(value['mae_s'],'.6f')+' s' for mode,value in r['metrics'].items())+'. '
text+='El criterio temporal predeclarado '+('se cumplio' if r['passed'] else 'no se cumplio')+'.\n\n'
text+='Se reprodujeron independientemente los cuantiles de los 10.000 viajes de las tres versiones y 65 consultas de cinco campos, incluyendo orden inverso y consultas individuales. '
text+='Se conservan todas las predicciones, metricas, configuraciones y fuentes, tambien cuando empeoran 131,06 s. '
text+=('Se retienen los pesos y referencias compactas para preparar el experimento primario congelado.' if r['passed'] else
 'Tras la reproduccion se retiraron solo los tres pesos transitorios y el indice de fuentes especifico de este piloto; su constructor y manifiesto permiten reconstruirlo.')
text+='\n\nEstas cifras corresponden a Train reservado, no a la validacion primaria de 127.941 viajes. El mejor MAE primario verificado sigue en 129,990034 s; el objetivo de confirmacion de 73,5 s sigue pendiente y Test cerrado.\n'
for path in [BASE/'RESUMEN_RESULTADOS.md',BASE/'ARQUITECTURA.md']:
 old=path.read_text(encoding='utf-8-sig');assert marker not in old;path.write_text(old+'\n\n'+text,encoding='utf-8')
print(json.dumps(dict(complete=True,practical_gate=bool(r['passed']),metrics=r['metrics'],retired_MiB=retention['retired_bytes']/1024**2,
 preserved_weight_n=len(retention['preserved_weights']),primary_best_unchanged=True,Test='SEALED')))
