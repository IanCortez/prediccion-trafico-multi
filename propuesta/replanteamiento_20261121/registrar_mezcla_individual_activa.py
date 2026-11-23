"""Register only observed live training and separately validated numeric reference."""
from base_replanteo import *
run=OUT/'mezcla_individual_mediana_Train_s0'
protocol=read(run/'protocolo.json');data=read(run/'verificacion_datos.json');gpu=read(run/'preflight_GPU.json')
proof=read(OUT/'verificacion_mediana_control_arco_v3/resultado.json')
stable=read(OUT/'verificacion_cuantiles_balanceados_s0/resultado.json')
assert data['passed'] and gpu['passed'] and proof['passed'] and stable['passed']
state=read(OUT/'CONTINUIDAD_OPERATIVA.json')
state['current_trial']=dict(experiment=run.name,pid=37484,exec_session=19856,log='entrenar_mezcla_individual_Train.log',
 stage='Threefreshmatchedarms,80fixedepochs each; observedsourcearm pending. No primaryValidation/Test.',original_solver_unchanged_during_run=True)
state['next_after_result']=[
 'Poll actual37484/session19856; never restart for an observationtimeout. Do not edit active/queued training/model/data/token modules.',
 'After all3arms terminal, run reproducir_mezcla_individual_Train.py alone on GPU. Verify original10000quantiles/raw65 and evaluate sameweights with stable signedlogtail solver.',
 'Numericalstress and stable solver independently passed before observedarmoutcome. Preserve originalFP32 outputs and stablevariants; never select precision by heldMAE.',
 'Inspect robust practicalgate and train/held generalization gap. Current no-source150.53s and geometry148.30s are completedheldTrainresults, not primaryValidation.',
 'If practicalgate fails with >20s generalizationgap, plan_escala_datos_Train.json defines951718fitrows and4full epochs (~14872updates vs12560old); activate only after independentreplay. No sourceK/grid/threshold sweep.',
 'Keep all completedmetrics/predictions/config and sources, even MAE>131.06. Finalpilotweights transient pendingreplay/decision. Bestprimary129.990034, qualifiedlineage and TransferTraj protected,73.5unmet,Testsealed.'
]
state['resource_policy']='OneGPU .5cap, actualresident50000input/B256 two-step preflight819.13MiB; no persistent rawtoken cache or extraGPSbank.'
state['current_individual_kernel_preflight']=gpu
state['numeric_reference_verification']=dict(file='verificacion_cuantiles_balanceados_s0/resultado.json',source_sha256=stable['sources']['cuantiles_mezcla_balanceada.py'],
 semantics='Stable reference recovers symmetric deepgap median300s; active naiveFloat32 gave137.83s. Location/scale derivatives logdensity normalized; only extreme logweight derivative bounded1e6sec.')
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
active=read(OUT/'arquitectura_activa.json')
active.update(current_experiment=run.name,current_protocol=str((run/'protocolo.json').relative_to(OUT)),
 checkpoint_kind='Fresh4member x72component individual-source mixture; Train40k/held10k, fixed80epoch controls. Originalprotectedbest unchanged.')
write(OUT/'arquitectura_activa.json',active)
p=BASE/'ARQUITECTURA.md';s=p.read_text(encoding='utf-8');marker='<!-- ejecucion_mezcla_individual_Train -->'
if marker not in s:
 s+='\n\n'+marker+'\n## Mezcla individual: implementación y controles activos\n\n'
 s+='La implementación conserva 64 componentes históricos y ocho generales por cada uno de cuatro miembros. Cada fuente usa 34 variables por distancia de arco y cuatro variables de compatibilidad con dos rutas públicas. El entrenamiento calcula la mediana de la mezcla conjunta de los cuatro miembros, coincidiendo con la regla de consulta, y añade 0,02 de regularización mediante probabilidad del tiempo original. No se promedian medianas ni se utiliza un padre neuronal que haya visto la reserva Train.\n\n'
 s+='Las diferencias finitas verificaron el gradiente, las probabilidades de los intervalos sumaron uno y añadir cinco puntos GPS estacionarios solo cambió las dos variables de tiempo observado. El control geométrico completo permaneció idéntico. También se comprobó el marco geométrico al rotar y escalar una trayectoria sintética. La prueba con todos los datos residentes y dos pasos reales de gradiente alcanzó 819,13 MiB en GPU, por debajo del límite fijado.\n\n'
 s+='Una prueba adicional detectó pérdida de precisión de la mediana al separar mucho dos modos estrechos. Se implementó por separado una referencia que compara colas en escala logarítmica y doble precisión; recuperó la mediana analítica de 300 s donde la evaluación directa en float32 devolvía 137,83 s. Sus derivadas de localización y escala también coincidieron con valores analíticos. El gradiente extremo de los pesos se limita explícitamente por estabilidad. Este cálculo no modifica el entrenamiento activo; se evaluarán sus mismos pesos congelados con ambas reglas y se conservarán ambos resultados. La base matemática de la diferenciación implícita es Figurnov et al., NeurIPS 2018: https://proceedings.neurips.cc/paper_files/paper/2018/file/92c8c96e4c37100777c7190b76d28233-Paper.pdf. No se presenta ese cálculo como una contribución nueva.\n\n'
 s+='Se mantienen 80 épocas fijas para las tres versiones. Los primeros controles revelan sobreajuste: sin fuentes se obtuvo 150,53 s y con tiempos geométricos 148,30 s en los mismos 10.000 viajes Train reservados, aunque el error de ajuste fue mucho menor. Está definido un ensayo posterior condicionado a ese diagnóstico: 951.718 viajes existentes anteriores al corte, cuatro épocas completas y un número de actualizaciones comparable, con la misma reserva y sin ampliar las variables de consulta. Solo se activará después del cierre y reproducción independiente del ensayo actual.\n'
p.write_text(s,encoding='utf-8')
print(json.dumps(dict(recorded=run.name,preflight_MiB=gpu['peak_allocated_MiB'],stable_reference_verified=True,best_primary_unchanged=True)))
