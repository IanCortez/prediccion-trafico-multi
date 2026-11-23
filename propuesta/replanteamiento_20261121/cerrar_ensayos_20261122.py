"""Close only completed trials; retain every metric/prediction regardless of MAE."""
from base_replanteo import *
access=deny_test()
run=OUT/'piloto_landmarks_red_Train_v2_s0'
r=read(run/'resultado.json')
assert r['complete'] and not r['passed'] and not r['weights']
assert sha(run/'predicciones_held_Train.npz')==r['predictions_sha256']
for name,digest in r['sources'].items():
 assert sha(run/'fuentes'/name)==digest
with np.load(run/'predicciones_held_Train.npz') as z:
 assert len(z['true'])==10000
 for name,reported in r['metrics'].items():assert metric(z['true'],z[name])==reported
 assert metric(z['true'],z['reference'])==r['reference']
conclusion=dict(complete=True,scope='Reused chronological heldTrain only; not primaryValidation or confirmation.',
 directed_network_mae_s=r['metrics']['directed_network']['mae_s'],
 euclidean_mae_s=r['metrics']['euclidean']['mae_s'],
 comparisons=r['comparisons'],passed=False,
 interpretation='Directed topology profiles did not outperform the matched Euclidean representation; no evidence for an additional useful topological effect in this pilot.',
 decision='Do not launch a full primary trial or sweep landmark count/tree depth.',
 all_metrics_predictions_config_sources_retained=True,Test='SEALED')
write(run/'conclusion.json',conclusion)
p=BASE/'RESUMEN_RESULTADOS.md';s=p.read_text(encoding='utf-8')
s=s.replace('42 checkpoints neuronales','43 checkpoints neuronales')
old='El ensayo conjunto de 24 épocas terminó en 130,685980 s y quedó conservado con reproducción completa. El ensayo activo corrige únicamente el prior geométrico auxiliar: conserva las 34 variables temporales, normalización, arquitectura e inicialización, durante seis épocas fijas.'
new='El ensayo conjunto de 24 épocas terminó en 130,685980 s. La corrección aislada del prior geométrico, con las mismas 34 variables e inicialización y seis épocas fijas, terminó en 129,994490 s: no supera el mínimo de 129,990034 s. Ambos checkpoints cumplen el umbral de conservación y pasaron reproducción completa.'
s=s.replace(old,new)
marker='<!-- resultado_landmarks_red_Train_v2_s0 -->'
if marker not in s:
 s+='\n\n'+marker+'\n## Distancias dirigidas a referencias públicas: piloto cerrado\n\n'
 s+='En una reserva cronológica reutilizada de Train de 10.000 consultas, las distancias a 32 referencias de la red obtuvieron 132,424078 s; el control con referencias euclidianas obtuvo 132,286900 s. La diferencia fue +0,137178 s, intervalo exploratorio por día [−0,399601; 0,661094]. No pasó el criterio predeclarado y no se escala a un experimento primario. Se conservan ambas predicciones, métricas, configuración, fuentes y verificaciones de consulta cruda. Estas cifras no corresponden a Validation primaria ni a Test.\n'
p.write_text(s,encoding='utf-8')
p=BASE/'ARQUITECTURA.md';s=p.read_text(encoding='utf-8')
s=s.replace('y 41 checkpoints neuronales','y 43 checkpoints neuronales')
s=s.replace('**Ensayo activo:** se conserva','**Ensayo cerrado: 129,994490 s, sin superar el mínimo vigente.** Se conserva')
if marker not in s:
 s+='\n\n'+marker+'\n## Referencias topológicas públicas: piloto cerrado\n\n'
 s+='Se codificaron cuatro perfiles dirigidos hacia y desde 32 cruces públicos de referencia; se contrastaron con la misma cantidad de variables euclidianas. El piloto usó solo Train y mantuvo los cinco campos públicos de consulta. MAE: 132,424078 s frente a 132,286900 s del control, sin cumplir el criterio de mejora. No se inicia una búsqueda de cantidad de referencias ni de profundidad de árboles.\n'
p.write_text(s,encoding='utf-8')
state=read(OUT/'CONTINUIDAD_OPERATIVA.json')
state['current_trial']=None
state['last_closed_trial']=dict(id=run.name,scope=conclusion['scope'],passed=False,metrics=r['metrics'])
state['last_primary_trial']=dict(id='prior_arco_subtrayectos_s0',validation=read(OUT/'prior_arco_subtrayectos_s0/resultado.json')['validation'],independent_replay=read(OUT/'prior_arco_subtrayectos_s0/reproduccion_independiente.json')['passed'])
state['next_after_result']=[
 'All current primary and Train-pilot trials closed; inspect actual processes before launching another heavy job.',
 'Retain all completed metrics/predictions including MAE above131.06, plus every qualified checkpoint and required lineage; TransferTraj unchanged.',
 'Retire further subtrajectory geometry/memory and landmark-count/depth variations after failed practical gates.',
 'Reconsider a structurally different hypothesis using Train-only feasibility evidence before another full primary trial; no Test access and no 73.5 claim.'
]
state['resource_policy']='One heavy GPU job at a time; no heavy job currently active. All files/execution inside E:/UTEC/Tesis/TesisIan with local Conda.'
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
state=read(OUT/'arquitectura_activa.json')
state.update(current_experiment=None,current_protocol=None,last_closed_experiment=run.name,last_primary_result=read(OUT/'prior_arco_subtrayectos_s0/resultado.json')['validation'])
state['checkpoint_kind']='Best retained: original time-sampled subtrajectory branch and required parent lineage; corrected auxiliary-prior branch separately preserved.'
write(OUT/'arquitectura_activa.json',state)
policy=read(OUT/'politica_conservacion.json')
assert policy['all_completed_experiments_including_MAE_above131_06']
write(OUT/'cierre_conservacion_20261122.json',dict(topology_pilot=conclusion,policy_confirmed=True,
 all_completed_results_including_worse131_06=True,closed_primary_replays=True,goal='ACTIVE',Test='SEALED'))
print(json.dumps(dict(closed=run.name,directed_network_mae_s=conclusion['directed_network_mae_s'],all_completed_results_preserved=True)))
