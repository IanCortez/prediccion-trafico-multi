"""Conclude matchedneuralglobalroute study withoutexpandingfailedgates."""
from base_replanteo import *
run=OUT/'piloto_posterior_global_neural_train_s0';r=read(run/'resultado.json');assert not r['passed']
file=run/'predicciones_held_Train.npz';assert sha(file)==r['predictions_sha256'];z=dict(np.load(file))
for mode in ['time_only','semantic','random']:assert metric(z['true'],z[mode])==r['metrics'][mode]
removed=[]
for mode in ['time_only','semantic','random']:
 p=run/('predicciones_'+mode+'.npz');q=dict(np.load(p))
 for key in ['row','timestamp']:assert np.array_equal(q[key],z[key])
 assert np.array_equal(q['quantiles'],z[mode+'_quantiles'])
 removed.append(dict(file=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size));assert p.resolve().is_relative_to(run.resolve())
write(run/'limpieza_duplicados.json',dict(removed=removed,allarrays_retained_in='predicciones_held_Train.npz'))
for entry in removed:(ROOT/entry['file']).unlink()
weights=read(run/'pesos_descartados.json');assert len(weights['removed'])==3 and all(not (run/q['file']).exists() for q in weights['removed'])
conclusion=dict(complete=True,metrics=r['metrics'],reference=r['reference'],comparisons=r['comparisons'],
 decision='Failedreferencegate: semantic141.891seconds versusstrong133.355reference. No fullprimaryneuralglobalposterior andnohiddenwidth/classcount/auxweighttuning.',
 evidence='Semantic−random1.7685seconds, butsemantic−timeonly.591seconds CIcrosses0; semantic remains8.5356secondsworsethanstrongreference.',
 limits='Signalversusrandomlabels insufficient toshowusefulgain. Fixed100epochsmallTraincohort experiment, not disproof ofallsemanticroute learning.',
 initial_weights_same=r['all_branches_same_initial_weights_batches'],all_metrics_quantiles_predictions_preserved=True,
 next='Finishindependenttreeglobalposteriorwithmatchedcontrols; CPUimplementationretiredforcost, scalarGPUimplementationactive.',
 scope=r['scope'],Test='sealed',goal='ACTIVE')
write(OUT/'conclusion_posterior_global_neural.json',conclusion)
s=read(OUT/'CONTINUIDAD_OPERATIVA.json');s['current_trial']=dict(experiment='piloto_posterior_ruta_global_gpu_train_s0',stage='OneGPU25percentmemory;9scalarquantiles perbranch, matchedsemantic/random andpublic-only controls. Fixedconfigs, noheldscoresyet.',pid=34568,session_id=82292,log='piloto_posterior_ruta_global_gpu_train.log',Test='sealed')
s['completed_new_instruments']['global_route_neural']=conclusion
s['retired_CPU_global_study']=dict(file='piloto_posterior_ruta_global_train_s0/retiro_por_recursos.json',restart=False,completed_public_stage='piloto_posterior_ruta_global_train_s0/resultado_etapa_publica.json')
s['next_after_result']=['FollowGPUscalarquantileprocess/session82292, immutablefrozenrunner/helper, nootherheavyGPU.',
 'Do not restartretiredCPUstudy. Itscompletedpubliconlystagepredictionsarepreserved, fullsemanticcomparisonunfinished.',
 'GPUsemantic>=2secondsreference+>=1secondrandom+notworsepublic; alternativelypublic-only>=2secondsreference, upperCI<0. No gatesoverruled.',
 'NegativeNNglobalstudyclosed141.891vs133.355; allpoint/quantilepredictionskept, threeunqualifiedweightsandexactduplicatearchivesremoved.',
 'Currentbest130.158,40qualifiers, originals andTransferTrajprotected, Testsealed;73.5goalactive.']
write(OUT/'CONTINUIDAD_OPERATIVA.json',s);print(json.dumps(dict(concluded_neural=True,metrics=r['metrics'],all_completed_predictions_preserved=True)),flush=True)
