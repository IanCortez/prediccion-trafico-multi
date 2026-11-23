"""Record the actual live bounded Train pilot and its falsifiable transfer hypothesis."""
from base_replanteo import *
run=OUT/'piloto_transferencia_intervalos_Train_s0'
plan=read(run/'plan.json');proof=read(run/'verificacion_publica.json')
assert proof['passed'] and proof['raw17_feature_maxerror']==0
state=read(OUT/'CONTINUIDAD_OPERATIVA.json')
state['current_trial']=dict(experiment=run.name,stage='Train-only113830observedintervals; publicfeatureinstrumentpassed; densityweighting plus matchedpseudo-labelcontrol.',
 pid=38228,exec_session=80329,log='piloto_transferencia_intervalos_Train.log',Test='SEALED')
state['previous_goal_turn_classification']='Progress: audited completed primary corrections and topology pilot, verified253outputs/36archives plus10TransferTraj and protected lineage.'
state['next_after_result']=[
 'Inspect actual live process/result before waiting or restarting; observationtimeouts alone are not failure.',
 'Require5s gain versus both baseline and storedreference,2s versus pseudo-labelcontrol, all dayCIupper<0. No primaryValidation trial if gatefails.',
 'Keep every completed metric/prediction/config/source regardless of131.06. Preserve qualified primary models and original/best lineage; Testsealed.',
 'Do not repeat warm-start auxiliary head, sourceweight/count/tree sweeps, geometry-prior variants or landmark-count variants.'
]
state['resource_policy']='OneGPU .25cap/CPU2 for boundedTrainpilot; sourcefeatures RAM-only, persistentpilotartifacts<150MiB.'
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
state=read(OUT/'arquitectura_activa.json')
state.update(current_experiment=run.name,current_protocol='piloto_transferencia_intervalos_Train_s0/plan.json',
 checkpoint_kind='Protectedbest unchanged; newpilot tests boundary-aware density-weighted observedintervalaugmentation with independent pseudo-labelcontrol.')
write(OUT/'arquitectura_activa.json',state)
p=BASE/'ARQUITECTURA.md';s=p.read_text(encoding='utf-8');marker='<!-- piloto_transferencia_intervalos_Train_s0 -->'
if marker not in s:
 s+='\n\n'+marker+'\n## Transferencia de intervalos a viajes completos: piloto Train\n\n'
 s+='La supervisión auxiliar anterior actualizó un codificador preentrenado con una cabeza separada y un peso fijo; no mejoró el resultado. Este piloto cambia el mecanismo: aprende particiones compartidas de árboles desde cero con 113.830 intervalos exactos de los mismos 40.000 padres de ajuste. Distingue si cada extremo coincide con el del viaje padre, corrige de forma acotada las diferencias de covariables mediante clasificación de dominio fuera de grupo, y reserva el 20 % de peso relativo para las fuentes. La consulta de un viaje completo siempre lleva ambos indicadores de extremo activos; no aporta longitud de secuencia ni duración.\n\n'
 s+='El control conserva exactamente los ejemplos, indicadores y pesos auxiliares, sustituyendo sus tiempos observados por predicciones del modelo sin aumento. Así se separa la información de los tiempos históricos de la regularización por más ejemplos. La corrección de densidad no garantiza eliminar diferencias de comportamiento entre un intervalo y un viaje completo; esa es precisamente la hipótesis sometida a prueba.\n\n'
 s+='Se evalúan las mismas 10.000 consultas cronológicas reutilizadas de Train. Antes de ajustar, las 206 variables públicas coincidieron exactamente con las existentes en 17 consultas crudas, también al invertir el orden y consultar un solo caso; se comprobó que la geometría vial no depende del perfil histórico excluido. No se utiliza la trayectoria de ninguna consulta reservada. Se exige una mejora de al menos 5 s frente al modelo sin aumento y la referencia almacenada, y 2 s frente al control con tiempos artificiales, con intervalos exploratorios por día favorables. Si falla, no se busca otro peso ni cantidad de intervalos.\n\n'
 s+='La motivación del submuestreo aparece en ProbETA, publicado en IEEE TITS 2025 y con revisión de autores de enero de 2026: https://arxiv.org/html/2407.05895v2. Ese modelo utiliza recorridos y viajes completados como contexto; este piloto OD es una adaptación propia y no una reproducción comparable de sus resultados.\n'
p.write_text(s,encoding='utf-8')
print(json.dumps(dict(recorded=run.name,public_instrument_passed=True,source_n=proof['auxiliary_n'],goal='ACTIVE')))
