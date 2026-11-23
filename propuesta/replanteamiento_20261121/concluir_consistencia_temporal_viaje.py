"""All-row audit and critical conclusion of the wholetrip-time experiment."""
from base_replanteo import *
access=deny_test();run=OUT/'consistencia_temporal_viaje_s0';bf=run/'etapa_B'
r=read(bf/'resultado.json');a=read(run/'resultado_modelo.json');assert r['complete'] and a['complete']
z=dict(np.load(bf/'validacion.npz'));d=dict(np.load(DATA/'validation_x.npz'));truth=np.load(DATA/'validation_y.npy')
assert len(truth)==127941 and np.array_equal(z['true'],truth) and sha(bf/'validacion.npz')==r['predictions_sha256']
for k in ['row','timestamp']:assert np.array_equal(z[k],d[k])
for name,m in r['metrics'].items():assert metric(truth,z[name])==m
for name,v in r['models'].items():
 if v['preserved']:assert sha(bf/v['file'])==v['sha256']
certfile=run/'certificacion_full246_MAE/resultado.json';cert=read(certfile) if certfile.exists() else None
if r['new_numerical_minimum']:assert cert and cert['passed']
decision=('Predeclaredgain passed; next matched8386parameterlocal-time-supervisioncontrol before attributing improvement toaggregateobjective.'
 if r['meaningful_gate_passed'] else
 'Aggregateknowntriptime calibration didnotpassprimaryusefulnessgate. Do not retuneepochs/scales/stylecounts; nextaddressglobalrouteuncertainty andtrain/queryprefixexposure.')
result=dict(complete=True,main_validation=r['main_validation'],comparisons=r['comparisons'],meaningful_gate_passed=r['meaningful_gate_passed'],
 new_numerical_minimum=r['new_numerical_minimum'],certification=cert,
 private_Train_before=a['private_Train_diagnostic_before'],private_Train_after=a['private_Train_diagnostic_after'],
 diagnostic_warning='PrivilegedTrainimprovement cannot substitute for primaryODValidation orconfirmation.',
 frozen_body_unchanged=a['frozen_body_unchanged'],decision=decision,
 no_primary_filter_label_or_query_information_change=True,all_completed_metrics_predictions_preserved=True,
 current_best_source='nuevo_mejor_verificado.json',goal='ACTIVE',Test='sealed')
write(OUT/'conclusion_consistencia_temporal_viaje.json',result)
state=read(OUT/'CONTINUIDAD_OPERATIVA.json');state['current_trial']=dict(experiment=run.name,stage='Complete; reviewcriticalconclusionandcontinue.',pid=None,Test='sealed')
state['latest_trip_consistency_experiment']=result
state['next_after_result']=[decision,'Ifnewnumericalminimumexists, certificatepassed; updatebestregistryandrecipehonestlybeforeothertraining.',
 'Keeporiginals, allqualifiedmodelsandTransferTraj; allcompletedmetrics/predictionsincludingabove131.06.',
 'Primaryconfirmation73.5unmetuntilindependentfrozencandidateevaluation; doNOTopenTestnow.']
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
p=BASE/'RESUMEN_RESULTADOS.md';s=p.read_text(encoding='utf-8-sig');title='### Consistencia del tiempo total por viaje'
assert title not in s
s+='\n'+title+'\n\nDos épocas fijas entrenaron solamente 8.386 parámetros de corrección positiva de tiempos; el cuerpo neuronal y el predictor de rutas permanecieron congelados. Se conservan todos los viajes y tiempos primarios.\n\n'
s+='MAE Validation de la propuesta: '+format(r['main_validation']['mae_s'],'.6f')+' s. Criterio de mejora útil: '+str(r['meaningful_gate_passed'])+'.\n\n'
s+='El diagnóstico con recorrido conocido de Train pasó de '+format(a['private_Train_diagnostic_before']['mae_s'],'.6f')+' a '+format(a['private_Train_diagnostic_after']['mae_s'],'.6f')+' s; es una supervisión privada y no equivale al MAE OD. Las métricas y predicciones completas se conservan incluso por encima de 131,06 s. Test cerrado; objetivo de confirmación 73,5 s pendiente.\n'
p.write_text(s,encoding='utf-8')
print(json.dumps(result),flush=True)
