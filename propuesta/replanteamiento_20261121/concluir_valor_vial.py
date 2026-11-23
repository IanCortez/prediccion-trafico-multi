"""Conclude completed primary roadvalue trial and register next raw5 gate."""
from base_replanteo import *
access=deny_test();run=OUT/'valor_vial_parcial_s0';r=read(run/'resultado.json');z=dict(np.load(run/'validacion.npz'));history=read(run/'historia.json')
assert sha(run/'validacion.npz')==r['predictions_sha256'] and metric(z['true'],z['pred'])==r['validation']
assert len(history)==r['epochs_run']==15 and r['best_epoch']==7 and not r['control_trigger_met'] and not r['saved_checkpoints']
assert all(not (run/name).exists() for name in ['ultima_reanudacion.pt','predicciones_avance.npz','mejor_temporal.pt'])
for epoch in range(16):
 assert z['epoch_'+str(epoch)].shape==(127941,3)
 if epoch:assert metric(z['true'],z['epoch_'+str(epoch)][:,1])==history[epoch-1]['validation']
dep={p.name:sha(p) for p in FROZEN.glob('*.py')}
write(run/'dependencias_fuentes_congeladas.json',dict(shared_protected_original_sources=dep,shared_conda_prefix=str(ROOT/'.conda-env'),
 original_feature_protocol_sha256=sha(BEST/'resultado.json'),street_bank_manifest_sha256=r['config']['bank_manifest_sha256'],
 auxiliary_manifest_sha256=r['config']['aux_manifest_sha256'],copying_large_data_or_env_not_required=True))
conclusion=dict(complete=True,validation=r['validation'],best_epoch=7,epochs=15,primary_n=127941,
 delta_vs_protected_best_s=r['validation']['mae_s']-r['config']['reference_validation']['mae_s'],paired=r['paired_vs_reference'],
 decision='Negative. No conditionalnoauxcontrol or architecturemicrotuning. Keep all16epochquantiles/metrics and selectedoutput; no unqualifiedweights. Best130.158 unchanged.',
 observation='Trainloss fell4.3807to4.0678 whileValidation bestepoch7 andlaterworse; progressauxiliary/localroadvalue did not translate to stronger ODmedian.',
 limits='No matchednoauxcontrol executed because usefulgain gatefailed. Cannot assignfailure causally to roadauxiliary or prove allstreetmodels fail.',
 next='Train-only raw5ODroutegeneration gate using learnedlocalpolicy; predicts fullroutes frompublicendpointsets, then time. Differentfrom amortizedglobalstreetvalue and privilegedteacherstateinstrument.',
 source_manifest_sha256=sha(run/'dependencias_fuentes_congeladas.json'),all_metrics_predictions_preserved=True,Test='sealed')
write(OUT/'conclusion_valor_vial.json',conclusion)
state=read(OUT/'CONTINUIDAD_OPERATIVA.json');state['current_trial']=dict(experiment='piloto_rutas_generadas_od_train_s0',stage='CPU2raw5ODroutegeneration/timegate; modelpolicyverifiedagainstnativeCatBoost, no heldactualroute supplied.',pid=None,log='piloto_rutas_generadas_od_train.log',Test='sealed')
state['completed_new_instruments']['primary_street_value']=dict(file='conclusion_valor_vial.json',MAE_s=r['validation']['mae_s'],epochs=15,selected_epoch=7,gain_gate_passed=False,qualified_models=0)
state['completed_new_instruments']['teacher_state_road_policy']=read(OUT/'piloto_politica_vial_train_s0/resultado.json')
state['next_after_result']=['Follow raw5generatedroute chronologicalTraingate; >=2secondreduction andupperCI<0 required beforefullprimarybank.',
 'Negativeamortizedstreetvalue134.779completed; no noauxcontrol and no retuning samebranch. Its all16epochquantiles retained.',
 'Do not equate94.024privilegedactualrouteoracle or90.82percentteacherstatenextroad accuracy with ODperformance.',
 'Ifraw5routegatepasses, prepareboundedpooledroutebank andprimarymodel; iffails, keepmetrics/predictions and move to a distinctmechanism.',
 'Best130.158,40previousqualifiers,originaldata andTransferTraj preserved; Testsealed and73.5goalactive.']
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
report=BASE/'RESUMEN_RESULTADOS.md';text=report.read_text(encoding='utf-8-sig')
old='Se ejecuta un diagnóstico cronológico solo en Train para medir el aporte de conocer la ruta real; su MAE no contará como predicción OD.'
new='El diagnóstico cronológico de Train obtuvo 94,024 s con ruta real frente a 133,355 s del control OD, resultado privilegiado que no cuenta para la meta. La nueva red de valor vial terminó 15 épocas en 134,779 s y se descartó; se conservaron todas sus predicciones. Sigue una prueba de rutas efectivamente generadas desde origen, destino y hora, sin rutas reales en la reserva.'
assert text.count(old)==1;report.write_text(text.replace(old,new),encoding='utf-8')
print(json.dumps(dict(concluded=True,validation=r['validation'],all_predictions_retained=True,unqualified_weights_absent=True,Test='sealed')),flush=True)
