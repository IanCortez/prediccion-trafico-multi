from base_replanteo import *
import ast
deny_test()
files=['generar_resumenes_progreso_recurrente.py','evaluar_progreso_recurrente.py','certificar_progreso_recurrente.py',
 'concluir_progreso_recurrente.py','verificar_conservacion.py']
for name in files:ast.parse((OUT/name).read_text(encoding='utf-8-sig'))
run=OUT/'progreso_vial_recurrente_s0';assert (run/'protocolo.json').exists()
sources={n:sha(OUT/n) for n in files}
write(OUT/'plan_flujo_progreso_recurrente.json',dict(sources=sources,
 orchestrator_sha256=sha(OUT/'continuar_flujo_progreso_recurrente.ps1'),Train_PID=30380,
 stages=['FinishsixfixedfullTrainepochs','Freezeepoch6','Public-onlyallTrain/Valrollouts','Fixedmatchedtreecomparison',
 'Independentcandidatecertificationifnewminimum','Criticalconclusion','Conservationaudit'],Test='sealed',goal='ACTIVE'))
resources={name:dict(sha256=sha(path),bytes=path.stat().st_size) for name,path in [
 ('public_D',DATA/'distancias_2014/D.npy'),
 ('static_graph',OUT/'banco_valor_vial_s0/grafo_estatico.npz'),
 ('corrected_adapter_manifest',OUT/'adaptador_progreso_vial_s0/resultado.json'),
 ('full_Train_teacher_manifest',OUT/'datos_calles_parciales_Train_s0/resultado.json')]}
write(run/'recursos_publicos_y_docentes.json',resources)
state=read(OUT/'CONTINUIDAD_OPERATIVA.json')
state['current_trial']=dict(experiment=run.name,stage='ACTIVE: fixedsixepochfullprimary/sequenceTrain, then frozenpublicrollouts+matchedtreecomparison.',pid=30380,
 full_primary_Train=1023485,auxiliary_Train=894105,auxiliary_tokens=41775036,Test='sealed')
state['recurrent_preflight']=read(OUT/'verificacion_progreso_recurrente.json')
state['recurrent_resource_batching']=read(OUT/'medicion_lotes_progreso_recurrente.json')
state['resource_policy']='OneheavyGPUmaximum. RecurrenttrainingB512, publicqueriesB1024; bothmeasured. Dnonpersistent,state8.55MiB. B1024teacherOOMpreservedandretiredbeforemetrics; noenvironmentchanges.'
state['next_after_result']=[
 'Continueentirefrozenpipeline; orchestratormonitorsactualTrainPID30380 andneverstartsheavyGPUstagesconcurrently.',
 'DoNOTeditactive/queuedmodel,data,trainer,rollout,treeevaluator,certifier/concluder sources. SHA-checkedbeforeexecution.',
 'If a stagefails, inspectactualprocess/log/statusbeforeanyresume; observationtimeoutisnotfailure. ResumeTrainonlyfromexactepochcheckpoint with--resume.',
 'Allcompletedmetrics/predictionsincluding>131.06; weightsbelow131.06+strictbests+dependencies protected.40historicalqualifiers andTransferTrajintact.',
 'Primarycohort/labels/queryfields unchanged; Testclosed; objective73.5notachievedyet.']
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
summary=BASE/'RESUMEN_RESULTADOS.md';s=summary.read_text(encoding='utf-8-sig')
title='### Corrección de posiciones y experimento recurrente en curso'
if title not in s:
 s+='\n'+title+'\n\nSe verificaron 41.775.036 tokens auxiliares y se corrigió la interpretación de 680.543 posiciones internas no iniciales y 689.590 transiciones que no recorren una calle comprimida completa. Los tiempos originales y todos los viajes primarios se conservan. No se atribuye a esta corrección una mejora de MAE antes de medirla.\n\n'
 s+='Se inició una alternativa con memoria del recorrido, cuatro estilos latentes y reloj previsto. Entrena seis épocas con 1.023.485 consultas primarias y secuencias auxiliares de 894.105 viajes. Las pruebas neuronales de llegada, inmovilización del estado al terminar, independencia por lote y reloj preciso pasaron. El lote de entrenamiento elegido usa unos 4 GB de memoria gráfica; el estado del modelo ocupa 8,55 MiB y excluye la matriz pública de distancias de 2 GB.\n\n'
 s+='Después se generarán recorridos solo a partir de OD y fecha para comparar el modelo con un control de igual capacidad. No hay todavía un resultado de MAE de esta alternativa. Test permanece cerrado. Se conservan métricas y predicciones también cuando superan 131,06 s.\n'
 summary.write_text(s,encoding='utf-8')
print(json.dumps(dict(frozen_pipeline_sources=sources,active_PID=30380,Test='sealed')))
