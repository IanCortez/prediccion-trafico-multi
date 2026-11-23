"""Audit the completed ordered-source time signal without overriding its declared expansion gate."""
from base_replanteo import *
run=OUT/'piloto_cdf_ordenada_Train_s0';r=read(run/'resultado.json')
assert r['complete'] and not r['passed'] and not r['weights']
assert sha(run/'predicciones_held_Train.npz')==r['predictions_sha256']
assert sha(run/'indices_fuentes.npz')==r['sourceindices_sha256']
for name,digest in r['sources'].items():assert sha(run/'fuentes'/name)==digest
with np.load(run/'predicciones_held_Train.npz') as z:
 for key,value in r['metrics'].items():assert metric(z['true'],z[key])==value
 with np.load(OUT/'oraculo_tiempo_calle_train_s0/predicciones_held_Train.npz') as ref:
  for key in ['row','timestamp','true']:assert np.array_equal(z[key],ref[key])
proof=read(run/'verificacion_fuentes_features.json')
assert proof['passed'] and proof['raw17_cached_summaries_exact'] and proof['reverse_singleton_exact']
assert sum(v['n'] for v in proof['statistics'])==50000
conclusion=dict(complete=True,metrics=r['metrics'],comparisons=r['comparisons'],declared5s_expansion_gate=False,
 observed_time_signal_vs_geometry_control_s=r['comparisons']['vs_geometry_control']['delta_mae_s'],
 decision='No fullprimarytrial for this CDF recipe; no K/grid/sourcecount/tree sweep and no posthoc threshold reduction.',
 interpretation='Historical source-time labels improve this reusedTrainpilot by about2.25s versus geometry-derived times. This is useful signal but below the declared5s expansion requirement versus baseline/reference.',
 retention='All completed metrics/predictions/config/exactsource/compactindices retained. Trainpilot130.955613 is not a qualifying fullprimaryValidation model under131.06; its weights were not persisted.',
 limit='No primaryValidation/confirmation MAE gain and no claim that73.5is reached. Small repeatedlyreusedTraininstrument, one modelseed.',
 proposed_next='Study information-preserving individual-source conditional distributions rather than compressedCDF summaries, first on Train with the same predeclared5s criterion. No primary trial without a new positive gate.',
 Test='SEALED',goal='ACTIVE')
write(run/'conclusion.json',conclusion)
p=BASE/'RESUMEN_RESULTADOS.md';s=p.read_text(encoding='utf-8');marker='<!-- resultado_cdf_ordenada_Train_s0 -->'
if marker not in s:
 s+='\n\n'+marker+'\n## Búsqueda conjunta: prueba temporal Train cerrada\n\n'
 s+='Con las fuentes fuera de la semana de cada consulta Train, las distribuciones de tiempos históricos obtuvieron 130,955613 s en 10.000 viajes Train. El modelo público sin memoria obtuvo 133,121328 s y el control de igual dimensión con tiempos derivados de geometría 133,210289 s. Las diferencias fueron −2,165715 s frente al modelo sin memoria, intervalo exploratorio por día [−3,120481; −1,221083], y −2,254676 s frente al control, intervalo [−3,285641; −1,276916].\n\n'
 s+='Existe señal temporal útil, pero no se cumplen los 5 s de mejora predeclarados para escalar este resumen a un experimento primario. No se cambia ese criterio ni se prueban nuevas cuadrículas o cantidades de fuentes. La cifra de 130,955613 s corresponde a Train reservado, no a los 127.941 viajes de Validation: no reemplaza al mejor modelo ni califica para conservar pesos por el umbral primario. Todas las predicciones, métricas, fuentes y referencias compactas se conservan. Test sigue cerrado.\n'
p.write_text(s,encoding='utf-8')
p=BASE/'ARQUITECTURA.md';s=p.read_text(encoding='utf-8')
if marker not in s:
 s+='\n\n'+marker+'\n## Señal temporal del nuevo buscador\n\n'
 s+='El instrumento temporal terminó en 130,955613 s en 10.000 consultas Train, frente a 133,121328 s sin memoria y 133,210289 s con el control geométrico de igual dimensión. La información de tiempos históricos aporta una diferencia exploratoria favorable de 2,25 s frente al control. No alcanza el criterio de 5 s para ampliar esta representación comprimida y no se evalúa en Validation primaria. El siguiente diseño debe conservar la relación individual entre geometría, contexto de extremos y tiempo de cada fuente, en lugar de perderla al resumir todo en distribuciones. Ese diseño requiere otra prueba Train con el mismo criterio exigente; no se reduce el umbral después del resultado.\n'
p.write_text(s,encoding='utf-8')
state=read(OUT/'CONTINUIDAD_OPERATIVA.json')
state['current_trial']=None
state['last_closed_timepilot']=conclusion
state['last_closed_timepilot_id']=run.name
state['next_after_result']=[
 'Both boundary-augmentation and orderedCDF Trainpilots terminal. No heavyjob active; verify actualCIM before nextlaunch.',
 'Orderedsource constructor passed independent sourceinterval/weight/top64, safepruning, raw65 and wholeparentweekexclusion checks. Geometriccoverage gain persisted underoriginalbbox.',
 'HistoricalCDF130.955613 vsbaseline133.121328 andgeometrycontrol133.210289 shows2.25s exploratorysignal, but predeclared5sgatefailed. No primarytrial or threshold reduction for this recipe.',
 'Next: inspect existing sourceattention model to avoid repetition; design individual-source conditional transfer preserving geometry/time/boundary relationships, Train-only with freshmatched controls and no heldlabel pretrainedparent.',
 'All completed predictions/metrics/configs including MAE>131.06 preserved; qualified43primarycheckpoints and TransferTraj protected. Bestprimary129.990034 unchanged;73.5notachieved;Testsealed.'
]
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
state=read(OUT/'arquitectura_activa.json')
state.update(current_experiment=None,current_protocol=None,last_closed_timepilot=conclusion,
 checkpoint_kind='Bestprimary129.990034 unchanged. Correctedorderedsource retrieval verified; compressedCDF pilotclosed, independent-source representation is next hypothesis.')
write(OUT/'arquitectura_activa.json',state)
print(json.dumps(dict(complete=True,metrics=r['metrics'],expansion_gate_passed=False,all_completed_evidence_preserved=True,new_primary_MAE_claim=False)))
