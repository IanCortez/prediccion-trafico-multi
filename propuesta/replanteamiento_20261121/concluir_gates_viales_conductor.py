"""Conclude negative Train-only hypotheses and remove exact discarded feature intermediates."""
from base_replanteo import *
access=deny_test();removed=[];entries=[]
specs=[
 ('piloto_rutas_generadas_od_train_s0','pred','descriptores_generados_Train.npz','features_sha256',
  'Publicraw5generatedroutesfailedthe>=2secondgate; no fullprimaryroutebank, no routinghyperparametersearch. TeacherstateaccuracydidnottranslatetoODtime.',
  'Generatedroutegeometrymean/std, not experimentpredictions.'),
 ('piloto_conductor_oraculo_train_s0','pred','etiquetas_conductor_Train.npz','hashed_labels_sha256',
  'True-driveroracleimprovement1.122seconds below3secondgate; no latentdriver training. DriverinputremainsforbiddeninpublicODqueries.',
  'RegenerablehashedTrainmetadata, not experimentpredictions.')]
for name,predkey,temporary,sha_key,decision,kind in specs:
 run=OUT/name;r=read(run/'resultado.json');assert not r['passed'];predfile=run/'predicciones_held_Train.npz';z=dict(np.load(predfile))
 assert sha(predfile)==r['predictions_sha256'] and metric(z['true'],z[predkey])==r['metrics'] and metric(z['true'],z['control'])==r['control']
 entries.append(dict(id=name,scope=r['scope'],metrics=r['metrics'],control=r['control'],comparison=r['comparison'],decision=decision,
  predictions_sha256=r['predictions_sha256'],all_metrics_predictions_sources_retained=True))
 p=run/temporary;assert p.resolve().is_relative_to(run.resolve()) and sha(p)==r[sha_key];size=p.stat().st_size
 removed.append(dict(file=str(p.relative_to(ROOT)),bytes=size,sha256=r[sha_key],kind=kind,recreate_from=str((run/'fuente_ejecutada.py').relative_to(ROOT))))
# Every scientificoutput verified before exact unlink; no activejob uses these twofiles.
write(OUT/'limpieza_gates_viales_conductor.json',dict(removed=removed,all_completed_metrics_predictions_preserved=True,originals_best_TransferTraj_unchanged=True))
for item in removed:(ROOT/item['file']).unlink()
write(OUT/'conclusion_gates_viales_conductor.json',dict(experiments=entries,not_primary_Validation=True,
 caution='True-route94.024 andteacherstate90.82percentaccuracy remainprivileged diagnostics, not deployableODachievements. Failedgenerateddescriptorconfiguration does not disproveallroutegenerators.',
 next='Independentlowranklocalroadtrafficfield usesactualTrainroadentryclock, sourceparentweekexcluded inallparams; onlypublicOD5fieldsatreprediction.',
 Test='sealed',goal='ACTIVE'))
state=read(OUT/'CONTINUIDAD_OPERATIVA.json')
state['current_trial']=dict(experiment='piloto_trafico_tensor_train_s0',stage='Sixsourceexcludedquarterhourroadpace/dwellfieldsfitted; fixed800tree ODgateinTrain only.',pid=22776,log='piloto_trafico_tensor_train.log',Test='sealed')
state['completed_new_instruments']['generated_route_OD_gate']=entries[0]
state['completed_new_instruments']['driver_oracle_gate']=entries[1]
state['next_after_result']=['CompletecrossfittrafficTrain gate; if>=1secondgain andupperCI<0, run matchedparentdepartureclockcontrol beforeprimarydesign.',
 'Ifnegative, keepmetrics/predictions and discard temporaryfields; no tuning rank/slot/timeoffsetjusttofind apositive.',
 'Do not expandgeneratedroutebank aftergatefailed; negativeprimaryamortizedstreetvaluealso closed.',
 'Best130.158and40qualifiersunchanged;240primaryoutputs in28archives audited. Allcompletedmetrics/predictionsincludingworse131.06preserved. Testclosed,73.5goalactive.']
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
print(json.dumps(dict(concluded=len(entries),removed_bytes=sum(x['bytes'] for x in removed),all_completed_predictions_retained=True)),flush=True)
