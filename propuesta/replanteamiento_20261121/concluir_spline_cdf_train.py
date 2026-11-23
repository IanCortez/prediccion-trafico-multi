"""Record negative flexible-CDF gate; goal remains active, Test sealed."""
from base_replanteo import *
access=deny_test();run=OUT/'piloto_spline_cdf_train_s0';r=read(run/'resultado.json');assert not r['passed'] and r['epochs']==50 and not (run/'modelo.pt').exists()
z=dict(np.load(run/'predicciones_held_Train.npz'));assert sha(run/'predicciones_held_Train.npz')==r['predictions_sha256']
assert metric(z['true'],z['pred'])==r['metrics'] and metric(z['true'],z['control'])==r['control']
history=read(run/'historia_Train.json');assert len(history)==50 and all(q['n']==200000 for q in history)
conclusion=dict(complete=True,scope=r['scope'],metrics=r['metrics'],control=r['control'],comparison=r['comparison'],epochs=50,
 decision='Failed>=1secondgate. NofullprimarySplineCDF and no bin/rank/epochsearch. Metrics, quantiles,protocol,sources andnormalization retained; no negativeweights.',
 observation=dict(initial_Train_loss=history[0]['loss'],final_Train_loss=history[-1]['loss'],held_evaluations=1),
 limits='Specificfixedbudgetarchitecture. Cannot concludeallnormalizingflowsfail or claima general irreducibleerrorbound.',
 next='Test globalroute-signatureposterior ratherthanlocalgreedydecisions, with matchedtime-onlycontrol andonlyrawODatprediction. Use existingTrainactualroute90features asauxiliaryteacher; no further routecount/histogramclocktuning.',
 Test='sealed',goal='ACTIVE')
write(OUT/'conclusion_spline_cdf_train.json',conclusion)
write(OUT/'siguiente_hipotesis_posterior_ruta.json',dict(status='Planned, no claimedresult.',
 hypothesis='Globalroutechoice may be coherent acrossmanyjunctions. Independentgreedy90.8percentlocalchoicesfailedODtime; learna whole-route latentposterior fromTraingeometricroute signatures.',
 existing_assets=['oraculo_tiempo_calle_train_s0/descriptores_ruta_Train.npz','oraculo_tiempo_calle_train_s0/auxiliares_Train.npz','banco_valor_vial_s0/train_proyecciones.npz','datos_calles_parciales_Train_s0/resultado.json'],
 prerequisite='Derivewhole-route semanticcodes fromfit-only geometryresidual versuspublicshortestroute, withoutusingT incodeconstruction.',
 query='Onlyoriginal5fields; routecode is aTrainauxiliarylabel, never actualheldroute input.',
 test='Chronological40kfit/10kheldsameoraclecohort; trainmarginalduration+routecodejointmodel andmatchedtime-onlycontrol. Heldroutecode onlydiagnostic, neverforwardinput. Fixedfinalevaluation, no heldparametersearch.',
 gate='Predeclarefullmodel andresourceboundsbeforeexecution; require >=2seconds gainoverexactCatBoostcontrol and >=1second overmatchedtime-onlymodel withupperdayCI<0 beforefullprimary.',
 fallback='Ifglobalrouteposterioralsofails, document inabilitytorecoverprivilegedroutegap and reevaluate informationavailable; never changeprimarylabels, filters orqueryinformation.',
 goal='73.5confirmationactive; best130.158 preserved, Testsealed.'))
state=read(OUT/'CONTINUIDAD_OPERATIVA.json');state['current_trial']=dict(experiment='global_route_signature_posterior_next',stage='PreviousSplineCDF50epochtrialcomplete/negative. Nextglobalposterior hypothesis requires operationalinstrument; no GPUjob currentlyclaimed.',pid=None,Test='sealed')
state['completed_new_instruments']['SplineCDF_train_gate']=conclusion
state['next_after_result']=['Continuegoalworkwithglobalrouteposterior instrument; distinguishcoherentwhole-routechoices fromgreedyteacherstateaccuracy.',
 'UseonlyfitTrainroutegeometryforteacherlabels andraw5queryfeaturesforprediction; matchedtime-onlycontrol, fixedheldevaluation.',
 'Do not rerunfailedquarterhourtrafficfields, driveroracle, generatedroute summaries, amortizedvalue, orSplineCDFbin/epochvariants.',
 'Keepbest130.158and40qualifiers, all240primaryoutputs/28archives andallTrainpilotmetrics/predictionsincluding>131.06.',
 'Goal73.5unmet; Testclosed. Do notmarkcompleteorblockedbecause experimentsarehard/negative.']
state['resource_note']='Exactdiscardedintermediates cleaned:47,805,443+27,632,418+22,641,856bytes plus6negativefieldweights. No completedpredictions/metrics deleted.'
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
report=BASE/'RESUMEN_RESULTADOS.md';text=report.read_text(encoding='utf-8-sig')
old='Continúa una familia distinta de distribuciones temporales mediante splines monótonos, con las mismas entradas y etiquetas.'
new='La variante de distribución mediante splines terminó 50 épocas y tampoco pasó el control interno (138,128 frente a 135,464 s). Sus métricas y predicciones se conservan. La investigación sigue activa; el siguiente contraste examina un posterior de rutas completas en vez de decisiones locales independientes, sin cambiar las entradas ni etiquetas.'
assert text.count(old)==1;report.write_text(text.replace(old,new),encoding='utf-8')
print(json.dumps(dict(concluded=True,metrics=r['metrics'],all_completed_predictions_retained=True,goal='ACTIVE',Test='sealed')),flush=True)
