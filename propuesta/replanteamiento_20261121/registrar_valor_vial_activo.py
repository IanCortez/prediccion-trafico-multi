"""Record gates, exact completion and active new primary architecture."""
from base_replanteo import *
access=deny_test();oracle=OUT/'oraculo_tiempo_calle_train_s0';r=read(oracle/'resultado.json');p=dict(np.load(oracle/'predicciones_held_Train.npz'))
assert sha(oracle/'predicciones_held_Train.npz')==r['predictions_sha256']
for key,field in [('control','control'),('privileged_route','privileged_route')]:
 assert metric(p['true'],p[field])==r['metrics'][key]
assert r['passed'];assert not r['Validation_access'] and not r['Test_access']
write(oracle/'ejecucion_verificada.json',dict(scientific_outputs_complete=True,metrics_predictions_verified=True,no_rerun_needed=True,
 terminal_exit_note='Only finalconsoleprint failed onndarrayserialization AFTER complete result/predictions/model werepersisted. No calculation failed, no sourceedit/retraining.',
 stdout_log='oraculo_tiempo_calle_train.log',source_sha256=r['source_sha256'],predictions_sha256=r['predictions_sha256'],scope=r['scope']))
corpus=read(OUT/'datos_calles_parciales_Train_s0/resultado.json');bank=read(OUT/'banco_valor_vial_s0/resultado.json');states=read(OUT/'estados_valor_vial_s0/resultado.json')
assert corpus['complete'] and states['complete']
write(OUT/'plan_valor_vial_amortizado.json',dict(
 decision='Use amortized streetremainingtimevalue rather than generating a fullroute atODinference in this trial.',
 reason='Avoid expensive or unstable path rollout while sharing learnedroadembeddings/valueheads with actualTrainprogress, localtime andnextroad auxiliaries. This is a distinctPortoadaptation, not publishedUNITEreplica.',
 evidence=dict(privileged_route_heldTrain_MAE=r['metrics'],delta=r['comparison'],partial_label_corpus=dict(trips=corpus['trips'],tokens=corpus['tokens'],trusted_time_fraction=corpus['trusted_time_fraction'])),
 architecture=dict(primary='Fresh4members, onehopgraphcontext over53766roads,16publicendpointpairs with2timecomponents each plus2globalfallback; discrete60..1785secondsupport.',
  input='Original5rawfields produce same216features asprotectedreference andstandalonepublic4mapprojections; no actualroute,N or elapsedtime.',
  auxiliary='Remainingvalue continuousNLL, partiallocaltimeHuber, neighbor/EOSCE on fixedtrustedTrainstates. No route decoder or observedlength in primaryprediction.',
  endpoint_uncertainty='Directedroadcandidates andpartialchainoffsets marginalised instead ofusingtruefullHMMendpoint.',
  source_paths='OnlyeligibleTrainroads used forauxiliarysupervision; no Validation/TestinteriorGPSdecoded.'),
 limits=['Actualrouteoracle94.02 is privileged and does not show howwellODcan infer route.','Some error is unseenroutechoice/dwell; no claim target73.5 is attainable by thisarchitecture.','Auxpartialtime allocatedbetween15seconds observations, not exactgroundtruthroadtime.','ReusedValidation/modelselection isexploratory, no independentconfirmation.'],
 control='Sameinitialweights/batches withoutaux; conditional on >=.5s primarygain andupperdayCI<0 aftercertification.',
 source='valor_vial_parcial_s0/fuentes',Test='sealed',goal='ACTIVE'))
path=OUT/'CONTINUIDAD_OPERATIVA.json';s=read(path);s['current_trial']=dict(experiment='valor_vial_parcial_s0',pid=15148,stage='GPUprimarytraining. CPU/GPUpreflights passed; initialuntrainedMAE179.1676s not a learnedresult.',log='entrenar_valor_vial.log',Test='sealed')
s['completed_new_instruments']['street_time_oracle']=dict(file='oraculo_tiempo_calle_train_s0/resultado.json',passed=True,control_s=r['metrics']['control']['mae_s'],privileged_route_s=r['metrics']['privileged_route']['mae_s'],not_primary=True)
s['bounded_new_street_assets']=dict(corpus_MiB=corpus['artifact_MiB'],public_bank_MiB=bank['artifact_MiB'],aux_states_MiB=states['artifact_MiB'],total_MiB=corpus['artifact_MiB']+bank['artifact_MiB']+states['artifact_MiB'],GPSrawcache=False)
s['next_after_result']=['Follow activefreshstreetvalue training, do not launch anotherGPUjob or edit frozen model/runner.',
 'Retainallcompletedprimarypredictions andeveryMAE<131.06candidate; best130.158 and40previousqualifiers remainprotected.',
 'Ifusefulgain>=.5s andupperCI<0, certifyraw5/order/singleton andmatchednoauxcontrol before claimingbenefit.',
 'Ifnegative, conclude fromloss/MAE androuteoraclegap; do not tune smallvariants blindly. Continue a genuinelydifferentnextmechanism toward73.5.',
 'Allprimaryfilters/labels/queryinputs/splits unchanged. Testsealed.']
write(path,s)
print(json.dumps(dict(recorded=True,active=s['current_trial'],aux_MiB=s['bounded_new_street_assets']['total_MiB'])))
