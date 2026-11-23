"""Jointclosure after all4fixedruns and exactreplays. Never lower gates or alterquerycohort."""
from base_replanteo import *
main=OUT/'puente_temporal_2d_Train_s0';control=OUT/'control_familia_puente_2d_Train_s0'
assert not (main/'conclusion.json').exists(),'One-shot retirement/closure; do not rerun.'
r=read(main/'resultado.json');c=read(control/'resultado.json')
assert r['complete'] and c['complete']
assert read(main/'reproduccion_independiente.json')['passed'] and read(control/'reproduccion_independiente.json')['passed']
a=read(main/'auditoria_resultado.json');assert a['complete'] and a['practical_gate_passed']==r['passed']
for folder,result in [(main,r),(control,c)]:assert sha(folder/'predicciones_held_Train.npz')==result['predictions_sha256']
best=read(OUT/'nuevo_mejor_verificado.json')
assert best['validation']['mae_s']==129.99003446901307 and sha(Path(best['model_file']))==best['model_sha256']
z=dict(np.load(control/'predicciones_held_Train.npz'));truth=z['true'];times=z['timestamp']
for key,other in [('Student_vs_matched_lognormal',z['pred']),('Student_vs_saved_publicNN',z['saved_publicNN'])]:
 p=paired(truth,z['Student_noaux'],other,times)
 assert abs(p['delta_mae_s']-c['comparisons'][key]['delta_mae_s'])<1e-12
secondary=all(p['delta_mae_s']<=-5 and p['ci95_s'][1]<0 for p in c['comparisons'].values())
assert secondary==c['secondary_head_gate_passed']
retention=dict(policy='All outcomes retained regardless131.06. PilotheldTrain does not qualifyweights by primaryValidationthreshold.',
 retired=[],retired_bytes=0,preserved=[],negative_weights_removed_only_after_exactreplay=True)
# The strongest prefix-only publiccontrol is a required, leak-free reference for the nextTraininstrument.
file=Path(c['model_file']).resolve();assert file.is_relative_to(control.resolve()) and sha(file)==c['model_sha256']
reference=dict(id=control.name,model_file=str(file),model_sha256=c['model_sha256'],bytes=file.stat().st_size,
 metrics=c['metrics'],scope='Reused10000heldTrain; NOT primaryValidation.',
 reason='ActivefixedTrain-only publicdensity reference for next roadstate instrument; no pretrainingonheldlabels.',
 inference_sources=c['sources'],constructor_sha256=c['constructor_sha256'],
 independent_replay_sha256=sha(control/'reproduccion_independiente.json'),qualifies_by_primary131_06=False,
 primary_best_not_replaced=True)
write(OUT/'referencia_publica_Train_red_vial.json',reference);retention['preserved'].append(reference)
for item in r['checkpoints']:
 file=Path(item['file']).resolve()
 assert file.is_relative_to(main.resolve()) and file.name=='epoca_4.pt' and sha(file)==item['sha256']
 record=dict(arm=item['arm'],file=str(file.relative_to(ROOT)),sha256=item['sha256'],bytes=file.stat().st_size)
 if r['passed'] or secondary:
  retention['preserved'].append(record)
 else:
  retention['retired'].append(record);write(main/'retencion_resultado.json',retention)
  file.unlink();retention['retired_bytes']+=record['bytes']
write(main/'retencion_resultado.json',retention)
r['independent_replay_pending']=False;r['independent_replay_sha256']=sha(main/'reproduccion_independiente.json');r['retention_record']='retencion_resultado.json'
c['independent_replay_pending']=False;c['independent_replay_sha256']=sha(control/'reproduccion_independiente.json');c['active_reference_manifest']='referencia_publica_Train_red_vial.json'
write(main/'resultado.json',r);write(control/'resultado.json',c)
a['secondary_matchedheadcontrol_pending']=False;a['retention_pending_secondary_control']=False
a['matched_lognormal_control_metrics']=c['metrics'];a['secondary_head_gate_passed']=secondary;write(main/'auditoria_resultado.json',a)
conclusion=dict(complete=True,main_auxiliary_gate_passed=bool(r['passed']),secondary_head_gate_passed=bool(secondary),
 main_metrics=r['metrics'],matched_lognormal_metrics=c['metrics'],main_comparisons=r['comparisons'],head_comparisons=c['comparisons'],
 all_four_models_independently_exactreplayed=True,
 critical_findings=[
 'Observedmotion supervision changes heldMAE by only-.72s versus noaux, with dayCIincludingzero; it is+.5355s worse than the linearcontrol. No declaredmain gate passed.',
 'The analyticStudent head withoutaux is1.2495s worse than the exactmatchedlognormal control; no declaredsecondary gate passed.',
 'Motioncorrelation exists in fitGPS, but that statisticalproperty and exactprobabilitycalculations do not imply an ODpredictiongain.',
 'A single relaxingAR regime perstyle and globalpathscale omit explicit roadconnectivity, destination-conditioned routechoice and discretewaitingstates. These are hypotheses for failure, not isolatedcausal conclusions.',
 'Continuousauxiliarylikelihood entropy differs across linear/observedtargets; a smallerauxNLL is not proof of betterdurationprediction.',
 'Same reusedheldTrain and oneinitialization; none of these numbers replaces the127941primaryValidation or126950unexposedconfirmation cohort.'
 ],
 next='Develop a bounded roadstate/kinetic transition instrument with an original15secondclock and explicitlymodelledwaiting, using rawfivefield publicattachments. First verify4096fitparents and resourcefeasibility; no rho/df/style/epoch sweeps.',
 reference_public_only=reference,retention=retention,all_metrics_predictions_logprobabilities_configs_andfrozen_sources_preserved=True,
 all_primary_rows_filters_andlabels_unchanged=True,no_extraqueryinformation=True,
 primary_best_unchanged_s=129.99003446901307,confirmation_goal_s=73.5,confirmation_goal_achieved=False,goal='ACTIVE',Test='SEALED',Validation=False)
write(main/'conclusion.json',conclusion)
write(control/'conclusion.json',dict(complete=True,secondary_gate_passed=secondary,metrics=c['metrics'],
 same_query_backbone_initialization_proven=True,independent_replay_passed=True,
 preserved_as_activeprefix_publicreference=True,primary_best_not_replaced=True,Test='SEALED'))
state=read(OUT/'CONTINUIDAD_OPERATIVA.json')
state['last_closed_timepilot_id']=main.name;state['last_closed_timepilot']=conclusion
state['current_trial']=dict(experiment='red_cinetica_15s_en_desarrollo',status='FITSTATE_CLOCK_AND_RESOURCE_INSTRUMENT_NEXT',live_worker=None,Test='SEALED')
state['secondary_head_control']=dict(status='COMPLETE_NEGATIVE_EXACTREPLAYED',id=control.name,metrics=c['metrics'],reference='referencia_publica_Train_red_vial.json')
state['next_after_result']=[
 conclusion['next'],
 'Do not rerun concludedStudenttraining or replay after its3weights intentionallyretired. All metrics/predictions/full116logprobs andsource/constructors preserved; retainedpubliclognormal prefixreference is available.',
 'Old map_matching_2014_v2/trayectorias_train intermediates absent aftercleanup. If roadstate labels needed, stream the existingoriginalpointbank through frozenpublicmapmatcher; never persist anotherfullGPS/mapmatchingbank.',
 'ReleasedUNITEsource audit is concreteCPUcontrolflow only: singletonETA differsfrommixedbatch underconstantsegmentpredictions. ODbranch hasfixed50cap and doesnot usequeryGTlength; speedprofile source isTrainonly. Preserve these distinctions.',
 'Fullprimarybest129.990034, all43qualifiedprimarymodels and10TransferTraj archives protected. Final73.5confirmationunmet,Testsealed.'
]
state['UNITE_source_audit']='auditoria_inferencia_UNITE_CPU_s0/resultado.json'
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
active=read(OUT/'arquitectura_activa.json');active['current_trial']=state['current_trial'];active['next']=state['next_after_result'];active['last_closed_trial']=main.name;active['secondary_head_control']=state['secondary_head_control'];write(OUT/'arquitectura_activa.json',active)
marker='<!-- resultado_puente_temporal_2d_Train_s0 -->'
paragraph=marker+'\n## Puente temporal bidimensional: comparación cerrada\n\n'
paragraph+='En las mismas 10.000 consultas cronológicas Train, tras cuatro épocas sobre 951.718 viajes anteriores al corte, los MAE fueron 130,872000 s sin objetivos auxiliares, 129,616500 s con movimiento lineal y 130,152000 s con movimiento observado. El control lognormal con la misma inicialización de codificadores y regla probabilística obtuvo 129,622500 s. Ninguna de las dos hipótesis predeclaradas cumplió su criterio.\n\n'
paragraph+='Se reprodujeron exactamente las probabilidades y los cuantiles de los cuatro modelos en los 10.000 viajes, además de 65 consultas públicas por reproducción, orden inverso y consultas individuales. No se necesitó el banco GPS original para inferir. Todas las métricas, predicciones, probabilidades, configuraciones y fuentes quedan conservadas. Se retiraron únicamente los tres pesos Student descartados; el control público queda como referencia de Train sin exposición a los viajes reservados.\n\n'
paragraph+='La correlación de movimiento observada en ajuste no produjo por sí misma una ganancia útil en duración OD. La siguiente alternativa incorporará explícitamente estados de la red vial y espera, manteniendo el reloj original de 15 s y comprobándolo antes de construir datos grandes. El mejor resultado primario sigue en 129,990034 s; no se ha demostrado el objetivo de confirmación de 73,5 s. Test permanece cerrado.\n'
for file in [BASE/'RESUMEN_RESULTADOS.md',BASE/'ARQUITECTURA.md']:
 old=file.read_text(encoding='utf-8-sig');assert marker not in old
 if file.name=='ARQUITECTURA.md':old=old.replace('Alternativa en ejecución: puente temporal de movimiento bidimensional','Alternativa evaluada: puente temporal de movimiento bidimensional')
 file.write_text(old+'\n\n'+paragraph,encoding='utf-8')
print(json.dumps(dict(complete=True,main_gate=r['passed'],secondary_gate=secondary,matched_control_MAE_s=c['metrics']['mae_s'],
 retired_MiB=retention['retired_bytes']/1024**2,active_publicreference_MiB=reference['bytes']/1024**2,
 primary_best_unchanged=True,all_outcomes_preserved=True,Test='SEALED')))
