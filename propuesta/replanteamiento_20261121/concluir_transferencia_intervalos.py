"""Close a completed negative Train augmentation result; preserve all evidence."""
from base_replanteo import *
run=OUT/'piloto_transferencia_intervalos_Train_s0';r=read(run/'resultado.json')
assert r['complete'] and not r['passed'] and not r['weights']
assert sha(run/'predicciones_held_Train.npz')==r['predictions_sha256']
assert sha(run/'procedencia_intervalos.npz')==r['provenance_sha256']
for name,digest in r['sources'].items():assert sha(run/'fuentes'/name)==digest
with np.load(run/'predicciones_held_Train.npz') as z:
 for name,m in r['metrics'].items():assert metric(z['true'],z[name])==m
 ref=dict(np.load(OUT/'oraculo_tiempo_calle_train_s0/predicciones_held_Train.npz'))
 assert np.array_equal(z['row'],ref['row']) and np.array_equal(z['true'],ref['true'])
conclusion=dict(complete=True,passed=False,metrics=r['metrics'],comparisons=r['comparisons'],
 scope='ReusedchronologicalTrainpilot; notprimaryValidation orconfirmation.',
 interpretation='Boundary-aware density-weighted sourceintervalaugmentation did not improve the freshbaseline; no evidence that source-time labels help through this recipe.',
 decision='Retire this augmentationrecipe without a sourceweight/count/tree sweep. No fullprimarytrial.',
 all_completed_predictions_metrics_config_sources_retained=True,weights_persisted=False,
 frozen_best_unchanged=True,Test='SEALED')
write(run/'conclusion.json',conclusion)
p=BASE/'RESUMEN_RESULTADOS.md';s=p.read_text(encoding='utf-8');marker='<!-- resultado_transferencia_intervalos_Train_s0 -->'
if marker not in s:
 s+='\n\n'+marker+'\n## Aumento de datos con transferencia de intervalos: cerrado\n\n'
 s+='El piloto con 113.830 intervalos de los mismos 40.000 padres de ajuste obtuvo 133,785328 s en 10.000 consultas cronológicas Train. El modelo sin aumento obtuvo 133,511432 s y el control con tiempos artificiales 133,971355 s. La diferencia frente al modelo sin aumento fue +0,273897 s, intervalo exploratorio por día [−0,382703; 0,935985]. No pasó el criterio predeclarado. No se escala ni se buscan otros pesos o cantidades de intervalos; todas las predicciones, métricas, configuración y fuentes se conservan.\n'
p.write_text(s,encoding='utf-8')
p=BASE/'ARQUITECTURA.md';s=p.read_text(encoding='utf-8')
s=s.replace('## Transferencia de intervalos a viajes completos: piloto Train','## Transferencia de intervalos a viajes completos: piloto cerrado')
if marker not in s:
 s+='\n\n'+marker+'\nEl aumento con ajuste de dominio terminó sin mejora: 133,785328 s frente a 133,511432 s sin aumento. Se descarta esa receta sin cambiar sus pesos o cantidad de ejemplos. El constructor ordenado ya pasó además una verificación independiente de posiciones y ordenación de las 64 fuentes, igualdad con y sin poda segura, consulta cruda y exclusión semanal previa al árbol. La nueva prueba temporal usa el banco original, sin aumento de datos: distribuciones de tiempos por tipo de extremo y un control de igual dimensión con tiempos derivados solo de geometría a 25 km/h.\n'
p.write_text(s,encoding='utf-8')
state=read(OUT/'CONTINUIDAD_OPERATIVA.json');state['last_closed_timepilot']=conclusion
state['completed_ordered_source_constructor']=read(OUT/'verificacion_fuentes_ordenadas_v2_s0/resultado.json')
state['current_trial']=dict(experiment='piloto_cdf_ordenada_Train_s0',stage='Train-onlycrossfit originalpointbank, correctedorderedsource retrieval; geometryCDFcontrol and freshbaseline.',
 exec_session=72815,log='piloto_cdf_ordenada_Train.log',Test='SEALED')
state['next_after_result']=[
 'Inspect actual process/log/result for piloto_cdf_ordenada_Train.py. Sourcefeatures are beingbuilt; oneGPU .25cap onlyafter all sourceproofs.',
 'Require5s improvement over freshpublic206baseline and storedreference plus2s over same-dimensional geometryCDFcontrol; all dayCIupper<0. No primarytrial justified solely by geometriccoverage.',
 'If positive, preservepilotmodels and perform fullunchangedprimaryproposal with raw5 independentreplay. If negative, retain metrics/preds/config and retire this CDFrecipe without grid/K/count sweeps.',
 'Do not rerun boundary/densityaugmentation: closed133.785328 versus133.511432baseline; all predictions preserved.',
 'Best129.990034 and all qualifiedweights/TransferTraj remain protected. Goal73.5 active and Testsealed.'
]
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
state=read(OUT/'arquitectura_activa.json')
state.update(current_experiment='piloto_cdf_ordenada_Train_s0',current_protocol='piloto_cdf_ordenada_Train_s0/plan.json',
 last_closed_timepilot=conclusion,checkpoint_kind='Protectedbest unchanged; Train-only historicalCDF signalgate after independentlyverified orderedsourceconstruction.')
write(OUT/'arquitectura_activa.json',state)
print(json.dumps(dict(closed=run.name,metrics=r['metrics'],all_completed_results_preserved=True)))
