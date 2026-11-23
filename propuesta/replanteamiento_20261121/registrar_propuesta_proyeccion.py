from base_replanteo import *
proof=read(OUT/'piloto_proyeccion_incierta_s0_v2/resultado.json');gpu=read(OUT/'mezcla_proyeccion_incierta_s0/verificacion_GPU_inicial.json');raw=read(OUT/'verificacion_entrada_proyeccion_raw.json')
assert proof['passed'] and gpu['passed'] and raw['passed']
plan=dict(hypothesis='Uncertain endpoint map attachment requires multiple plausible public routes, not only a single best attachment.',
 primary_cohort_unchanged=True,query=['o_lon','o_lat','d_lon','d_lat','timestamp'],
 model='Original8unplanned/time-geometry modes retained, plus8shared-head candidate time-geometry modes and a learned set-conditioned gate.',
 extra_hypotheses='Two protected defaults plus six alternatives from three additional projection pairs selected without labels.',
 Train_only_coverage_gate=proof,canonical_precision='FP16new structured inputs, original216inputs and modelFP32 unchanged.',
 initial_gate_logit=-4,initial_new_mixture_is_not_reference=True,
 protocol='3epochs frozen base plus6joint, fixed before results; conditional two-default-route control with identical initialization and learning protocol.',
 interpretation='Geometric oracle gain is coverage evidence, not ODtime MAE or a deployable selector. ReusedValidation and one seed remain exploratory.',
 Test='sealed',goal_confirmation_s=73.5)
write(OUT/'plan_mezcla_proyeccion.json',plan)
text="""

<!-- mezcla_incertidumbre_proyeccion -->
## Incertidumbre en la conexión al mapa

El piloto exclusivamente deTrain pasó el umbral fijado: en20000viajes, elegir con la geometría real entre las dos rutas protegidas y una recta da269,062286m de error medio; ampliar a ocho rutas más la recta baja a248,993851m. La diferencia20,068462m(7,46%) justifica ensayar más hipótesis. Un20,075% mejora más de10m. Es un selector oráculo para medir cobertura geométrica: no es un resultado deMAE ni una predicción disponible en consulta.

La arquitectura conserva los ocho componentes originales para cubrir rutas no representadas y añade ocho componentes de tiempo/geometría con una cabeza compartida. Dos hipótesis son las originales por longitud/tiempo libre. Las otras seis combinan ambos costes para tres parejas adicionales de proyecciones de origen/destino, elegidas por residuo GPS sin etiquetas. Una red compartida545-128-128-31recibe el estado global y las características de cada ruta; una puerta574-64-1usa el estado y los resúmenes del conjunto. La prioridad GPS usa30m, igual que el ajustador de mapa deTrain. La puerta empieza con logit-4: la nueva mezcla inicial tiene aproximadamente1,8%de masa candidata y no se confunde con la referencia original.

~~~mermaid
flowchart LR
 Q["Origen, destino y salida"] --> X["216 características originales"]
 Q --> R["8 rutas públicas con conexiones plausibles"]
 X --> B["Torre global protegida y 8 modos originales"]
 B --> H["Estado global compartido"]
 R --> C["Cabeza compartida de tiempo y geometría"]
 H --> C
 H --> G["Puerta según estado e incertidumbre del conjunto"]
 R --> G
 B --> M["Mezcla de 16 componentes"]
 C --> M
 G --> M
 M --> P["Distribución y mediana del tiempo"]
 T["Duración y geometría reales de Train"] -. "Solo pérdidas de entrenamiento" .-> M
~~~

La preparación retuvo todos los viajes originales y no leyó etiquetas devalidación. El nuevo banco ocupa362,75MiB. Solo sus entradas adicionales se almacenan enFP16 y se convierten aFP32; se mantienen las216entradas originales y las etiquetas. Media/desviación de rutas se ajustaron con las1.023.485filas elegibles deTrain antes de procesarvalidación. Los históricos excluyen la semana de cada consultaTrain a todos los niveles; para consultasholdoutsolo usanTrain.

Los instrumentosCPU comprobaron probabilidades normalizadas, recuperación del modelo original al quitar todas las rutas, gradientes finitos, invariancia del conjunto al permutar hipótesis del mismo tipo y ausencia de acceso a las rutas adicionales en el control de dos hipótesis. El instrumentoGPU reprodujo exactamente las127.941predicciones originales por el camino base y validó la pérdida conjunta con16componentes; el error máximo de normalización de responsabilidades fue3,58e-7. Desde65consultas con solo cinco campos se reconstruyeron exactamente las216entradas originales y el banco canónico; pasaron orden inverso, consulta individual y rechazo de trayectoria, cantidad de puntos, duración, tipo y procedencia.

Se fijaron tres épocas con la torre base congelada y seis conjuntas, con pérdidas originalesNLL+5RPS+0,2geometría. Solo si hay una reducción respecto de131,256712s se ejecutará el control con las dos rutas originales, conservando arquitectura, inicialización, normalización, muestreo y entrenamiento. Se mantiene la distinción entre una nueva función inicialmente distinta y una mejora aprendida. Guardar un candidato no acredita confirmación ni cercanía a73,5s.

Estado vivo: mezcla_proyeccion_incierta_s0.log y CONTINUIDAD_OPERATIVA.json. Todas las métricas/predicciones se conservan, también por encima de131,06s. Los pesos de todo nuevo mejor resultado y de cada candidato menor de131,06s quedan protegidos. Objetivo73,5s pendiente; Test cerrado.
"""
for dest in [OUT/'README.md',BASE/'RESUMEN_RESULTADOS.md']:
 old=dest.read_text(encoding='utf-8-sig');assert '<!-- mezcla_incertidumbre_proyeccion -->' not in old;dest.write_text(old+text,encoding='utf-8')
state=read(OUT/'CONTINUIDAD_OPERATIVA.json')
state['current_trial']=dict(experiment='mezcla_proyeccion_incierta_s0',pid=11480,exec_session_id=66149,
 stage='One GPU, fixed3frozen+6joint epochs, sources frozen',source_sha256=sha(OUT/'modelo_mezcla_proyeccion.py'),Test='sealed')
state['provisional_candidate']=dict(epoch=3,validation_mae_s=131.15787745914133,
 path='mezcla_proyeccion_incierta_s0/epoca_3.pt',status='Saved strict new best, still above131.06; not confirmation; needs complete run/control/reproduction')
state['resource_policy']='One heavy GPU;363MiBnew candidate cache bounded at500MiB; no diskCDFbank; clean only exact discarded intermediates.'
state['next_after_result']=['Verify actual process11480/session66149 before any action; never edit active trainer/model or imported sources.','Complete9epochs and assess all preserved point metrics. Epoch0 is new untrained mixture, not protected reference.','If any actual new function improves original reference, run defaults control only after expanded process truly terminal; verify equal initial state SHA.','Then freeze selected checkpoints and verify inference from rawfivefields and cached fullValidation. Keep Test closed until a genuinely strong candidate.', 'Preserve all completed metrics/predictions, every strict new best and fullMAE<131.06. Original best and TransferTraj unchanged.', 'Do not close goal after a batch;73.5confirmation still unmet.']
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
print('Updated architecture, instruments, preservation and active-trial state.')
