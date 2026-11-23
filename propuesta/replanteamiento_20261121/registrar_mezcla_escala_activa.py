"""Register actual all-data preflight/source proofs; keep primarygoal and sealedTest intact."""
from base_replanteo import *
run=OUT/'mezcla_individual_escala_Train_s0';source=read(OUT/'indices_escala_Train_s0/resultado.json')
gpu=read(run/'preflight_GPU.json');raw=read(run/'verificacion_stream_crudo.json');data=read(run/'verificacion_datos.json')
assert source['complete'] and source['held_all10000_refs_identical_to_smallpilot']
assert gpu['passed'] and raw['passed'] and data['passed']
assert gpu['initial_neural_sha256']==read(OUT/'mezcla_individual_mediana_Train_s0/resultado.json')['initial_neural_sha256']
state=read(OUT/'CONTINUIDAD_OPERATIVA.json')
state['current_trial']=dict(experiment=run.name,pid=26920,exec_session=37725,log='entrenar_mezcla_escala_Train.log',
 stage='Freshmatched none/geometry/observed,951718fit+10000unchangedheld,4fixedfullpasses. Actualstream/gradient/raw65 preflightpassed.',Test='SEALED')
state['previous_goal_turn_classification']='Progress:completed/replayed3smallNNarms, identifiedgeneralizationgap, retirednegativeweights, full951718sourceconstructor finished and streamtrainingstarted.'
state['next_after_result']=[
 'Monitor actual26920/session37725/log. OneheavyGPU. Do not edit current/queued runner, data, model, quantile or token imports.',
 'All-data sourcecache complete266.24MiB under350MiBcap; every held10000 source reference identical to oldpilot. Originalquerylabels/filters unchanged.',
 'Actualresidentall-data B256 preflight peak1961.16MiB. SourceRnormalization32768fitqueries only, publicnormalizers/timeanchors951718fitlabels only. No fullrawtokencache.',
 'Fixedfinal4epochs for all3arms; no heldstopping or epoch/grid/threshold sweeps. Everyfitrow once perfullpass,14872updates perarm.',
 'After actualterminalresult run reproducir_mezcla_escala_Train.py alone onGPU; requirefull10000quantiles/raw65/reverse/singleton checks before retention/promotion.',
 'Keep original>=5s matchedneuralbaseline/reference and>=2s geometryproxy gate. If passed, prepare fullunchangedprimary training. If failed, preserveallpreds/metrics/config/exactsources and critically reassess before any newexperiment.',
 'Originalbest129.990034 and43qualifiedprimarycheckpoints plus10TransferTraj predictions remain protected. Finalgoal73.5notmet; Testsealed.'
]
state['resource_policy']='OneGPU .5cap; measured1961.16MiB withresidentallX/sourceindices/pointbank. Dynamic38Dsource tokens perbatch, cache266.24MiB, no extraGPSbank.'
state['full_source_constructor']=dict(fit_n=source['fit_n'],held_n=source['held_n'],MiB=source['MiB'],sha256=source['indices_sha256'],heldreferences_identical=True)
state['current_all_data_preflight']=gpu
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
active=read(OUT/'arquitectura_activa.json')
active.update(current_experiment=run.name,current_protocol='mezcla_individual_escala_Train_s0/protocolo.json',
 checkpoint_kind='Freshstreamed4member x72component sourcekernel, all951718precutoffTrainfit, original10000held; no primarycandidate yet.')
write(OUT/'arquitectura_activa.json',active)
plan=read(OUT/'plan_escala_datos_Train.json');plan['status']='ACTIVE_AFTER_SMALLPILOT_INDEPENDENT_REPLAY_AND_SOURCE_CHECKS'
plan['sourceindices_manifest']='indices_escala_Train_s0/resultado.json';plan['actual_run']='mezcla_individual_escala_Train_s0/protocolo.json'
write(OUT/'plan_escala_datos_Train.json',plan)
p=BASE/'ARQUITECTURA.md';s=p.read_text(encoding='utf-8');marker='<!-- mezcla_individual_escala_Train_activa -->'
if marker not in s:
 s+='\n\n'+marker+'\n## Comparación ampliada: 951.718 viajes de ajuste\n\n'
 s+='La preparación de fuentes terminó y verificó las referencias de los 961.718 casos: 951.718 de ajuste y los mismos 10.000 reservados. Todos los padres fuente llegan antes del corte; los padres del grupo semanal de cada consulta de ajuste se excluyen antes de construir el árbol. Se comprobaron ausencia de fuentes propias, límites geográficos y tiempo del intervalo. Las referencias de los 10.000 casos reservados coinciden exactamente con las del piloto anterior. El archivo comprimido ocupa 266,24 MiB.\n\n'
 s+='El controlador completo entrena cuatro épocas por versión, 14.872 actualizaciones, con descenso cosenoidal por actualización. Conserva la misma arquitectura, semilla e inicialización neuronal del piloto pequeño. Cambian la cantidad de datos disponibles y el cálculo numérico estable previamente verificado. Las tres versiones nuevas comparten estos cambios y todos sus datos de ajuste; sus diferencias permiten estudiar el aporte de las fuentes históricas.\n\n'
 s+='Las 38 variables de cada fuente se generan en cada lote desde el banco de puntos y la distancia de arco existentes. No se guardan tensores de fuentes para toda la población. Las estadísticas públicas y anclajes se ajustan solo con los 951.718 viajes; las estadísticas de fuentes usan 32.768 consultas de ajuste seleccionadas con semilla fija. La comprobación real de gradientes y memoria pasó con 1.961,16 MiB; 65 consultas crudas reprodujeron exactamente las variables públicas y los tokens. Invertir el orden y consultar un caso individual mantuvo los resultados.\n\n'
 s+='No se consulta el error reservado hasta terminar las cuatro épocas de cada versión. Se conservan los criterios de mejora de 5 s frente al control sin fuentes y la referencia, y 2 s frente al control geométrico. Esta sigue siendo una reserva Train reutilizada, con carácter exploratorio. El mejor MAE primario continúa en 129,990034 s y Test permanece cerrado.\n'
p.write_text(s,encoding='utf-8')
print(json.dumps(dict(recorded_live=run.name,fit_n=source['fit_n'],peak_GPU_MiB=gpu['peak_allocated_MiB'],source_cache_MiB=source['MiB'],goal='ACTIVE')))
