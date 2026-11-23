from base_replanteo import *
best=read(OUT/'nuevo_mejor_verificado.json');assert best['id']=='atencion_subtrayectos_s0'
p=BASE/'ARQUITECTURA.md';s=p.read_text();a=s.index('## Mejor modelo verificado:');b=s.index('## Progreso vial recurrente:',a)
section="""## Mejor modelo verificado: 129,990034 s

El mínimo de Validation corresponde a atencion_subtrayectos_s0, época final fija 6. Incorpora ocho componentes de atención sobre subtrayectos a un padre congelado de 30 componentes, atencion_historia_individual_s0, época 3. Ese padre requiere historica_fresca_s0, época 27, de 22 componentes. Se conservan todos los antecesores.

La distribución final contiene 38 componentes y 116 tiempos entre 60 y 1.785 s. El bloque de 22 combina ocho modos generales, ocho hipótesis públicas de ruta y seis prototipos históricos. El siguiente bloque añade ocho componentes sobre 64 viajes completos; el último añade ocho sobre 64 intervalos ordenados de viajes históricos.

Cada intervalo aporta 34 variables: siete puntos geométricos, tiempo fuente, cercanía a los extremos de consulta, calendario y contexto de comienzo/final del viaje fuente. La normalización y el anclaje temporal se ajustan con 32.768 consultas Train; las fuentes excluyen todo el grupo semanal de cada consulta Train antes de construir el índice espacial. Validation solo consulta fuentes Train cuya llegada precede a su inicio.

~~~mermaid
flowchart LR
 Q["Cinco campos de consulta"] --> F["216 características y rutas públicas"]
 Q --> H["Viajes completos de Train"]
 Q --> S["Intervalos ordenados de Train"]
 F --> P["Mezcla de 22 componentes"]
 H --> P
 P --> A["Atención histórica: 30 componentes"]
 H --> A
 A --> B["Atención de subtrayectos: 38 componentes"]
 S --> B
 B --> T["Distribución y mediana previstas"]
~~~

La diferencia frente al padre de 130,158042 s es −0,168007 s; el intervalo exploratorio por día es [−0,222310; −0,110332]. Es un cambio pequeño y no cumple el criterio predeclarado de mejora práctica de 0,5 s. RMSE pasa de 209,353414 a 209,651248 s: empeora ligeramente. No se afirma una mejora causal ni confirmación independiente.

La reproducción recargó los pesos y comprobó los cuantiles de las 127.941 consultas, más 65 consultas construidas exclusivamente con cinco campos, inversión de orden y caso individual. La evaluación final se recuperó cargando solo Validation después de superar el límite de memoria al mantener Train residente; los pesos coincidían con los de la recuperación final. No se repitió el entrenamiento ni se amplió el límite.

Para reproducirlo se necesitan la rama de 0,90 MiB, sus dos antecesores, los bancos públicos e históricos, la memoria original de puntos y los índices compactos. Las dependencias y huellas están en replanteamiento_20261121/nuevo_mejor_verificado.json y modelos_atencion_subtrayectos_protegidos.json. La receta es replanteamiento_20261121/reproducir_atencion_subtrayectos.py. Se mantienen además los demás mínimos y 41 checkpoints neuronales que cumplen el umbral de conservación.

"""
s=s[:a]+section+s[b:]
s=s.replace('**Ensayo siguiente:** una rama de atención que aprende con Train el ajuste entre tiempos de subtrayectos y viajes completos.',
 '**Rama de atención ya evaluada:** aprende con Train el ajuste entre tiempos de subtrayectos y viajes completos; obtuvo el mínimo numérico de 129,990034 s descrito arriba.')
needle='[TrajRoute, MDM 2025]'
index=s.index(needle)
joint="""**Ensayo actual: entrenamiento conjunto desde cero.** La rama anterior aprendió con un padre que ya había visto todas las etiquetas Train. Se comprueba ahora si entrenar todo el sistema desde el inicio permite aprovechar mejor fuentes obtenidas fuera del grupo semanal. Esto es una hipótesis de optimización, no una causa demostrada del error.

Se inicializan de nuevo todos los pesos neuronales y se entrenan conjuntamente los 22 componentes de rutas/prototipos y ocho de subtrayectos. No se cargan pesos preentrenados. Se mantienen los mismos datos y fuentes y se fijan 24 épocas con descenso cosenoidal del paso de aprendizaje. La selección usa únicamente el estado final. Antes de entrenar se comprueban gradientes del bloque general y del nuevo experto, además de la memoria real de una consulta de 2.048 casos. Al evaluar se liberan los tensores Train. No se construyen bancos de distribuciones.

"""
if '**Ensayo actual: entrenamiento conjunto desde cero.**' not in s:s=s[:index]+joint+s[index:]
p.write_text(s,encoding='utf-8')
write(OUT/'revision_alternativas_20261122.json',dict(
 considered=[dict(title='TrajRoute',year=2025,url='https://www.johnkrumm.net/_files/ugd/3e3382_9d657170e9144dcb873791573d75c542.pdf',
  lesson='Followobservedtrajectoryintervals; paperalsoallowsstitching andmapfallback.',
  limitation='Routing vsAzureMaps, SanFrancisco, ±30minutequerywindow; notourevaluation orTrain-onlyhistoricaltraffic. No paperfilteradopted.'),
 dict(title='MDTI: MultimodalTrajectoryRepresentationLearningforTravelTimeEstimation',year=2026,url='https://arxiv.org/html/2510.05840v2',
  lesson='Modality-specificGPS/grid/roadencoders, contrastivealignment andmasking.',
  limitation='Consumesobservedtrajectory atquery. WouldrequireTrain-onlyteacher orODadaptation andnewproof; itsreportedTTEscoreisnotODbenchmark. No directreplicaclaim.'),
 dict(title='ETA PredictioninLast-MileLogisticsUnderDomainShift',year=2026,url='https://www.mdpi.com/2076-3417/16/17/8861',
  lesson='Targetsupervision andmatchedtargetonlycontrols matterfortransfer.',
  limitation='Lastmilelogistics andlabeledtarget adaptation differs; cannotconsumeValidation/Testlabels forourtraining.')],
 implementation='Orderedsubtrajectorysourceexpert completed129.990034; jointfresh30componentexperimentnext.',
 Test='sealed',goal='ACTIVE'))
print('Bestarchitecture and currentjoint-training rationale documented.')
