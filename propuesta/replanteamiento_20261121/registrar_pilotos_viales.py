"""Record completed gates and current bounded trial without modifying model sources."""
from base_replanteo import *
access=deny_test()
p=OUT/'CONTINUIDAD_OPERATIVA.json';state=read(p);neural=read(OUT/'verificacion_unite_neural_absorcion.json')
assert neural['passed']
state['current_trial']=dict(experiment='oraculo_tiempo_calle_train_s0',stage='ChronologicalTrain40kfit10kheld, matched206publicOD versus privilegedactualroute geometry. Not primaryOD. CPU2, fixed1000trees.',pid=27328,Test='sealed',log='oraculo_tiempo_calle_train.log')
state['completed_new_instruments']=dict(
 categorical=dict(file='piloto_categorias_od_train_s0/resultado.json',passed=False,delta_mae_s=-.5688103123235,reason='Belowpredeclared1secondgate; no fullprimarytrial.'),
 whole_trip_matching=dict(file='piloto_etiquetas_viales_train_s0/resultado.json',passed=False,zero_reset_fraction=.81494140625,reason='Failed90percent wholetripgate; remainsfailed.'),
 partial_step_trace=dict(file='diagnostico_resets_viales_s0/resultado.json',passed=True,trusted_step_fraction=.993350670665114,original_matching_exact=True),
 partial_time_labels=dict(file='piloto_cadenas_parciales_train_s0/resultado.json',trusted_time_fraction=.9914077631825066,primary_labels_exact=True,trips=4096),
 neural_absorption=dict(file='verificacion_unite_neural_absorcion.json',passed=True,real_time_components=True,trained_Portobenchmark=False,
  original_synthetic_single=3,original_synthetic_mixed=4,corrected_synthetic_single=4,corrected_synthetic_mixed=4))
state['next_after_result']=[
 'Do not repeatattention/historical/directmedian families; usefulgain gates alreadyfailed.',
 'Finish40kfit10kheldTrainstreettimeoracle. Necessary diagnostic>=3seconds withupperCI<0, not deployableOD score or bound.',
 'Iforaclepasses, build bounded streamedTrainauxiliary streetbank and raw5ODmodel with uncertainpublicendpoints, partialdistances, maskedGAPs and absorbingarrival.',
 'Iforaclefails, retainallmetrics/predictions and do not train this streetmodel merely because geometriccoveragepassed.',
 'Allprimaryqueries/labels/splits unchanged; no actualroute,N,timing,driver,calltype in publicAPI. NoTestdata.',
 'Preserveoriginal,bestrequiredparent,40qualifyingcheckpoints andallcompletedpredictionsincluding>131.06seconds; goal73.5 remainsactive.']
state['latest_storage_snapshot']=read(OUT/'inventario_conservacion.json')
state['resource_policy']='OneheavyGPU; currentpilotCPU2only. Lastsnapshot15.844GiB. Streetpilot200MiBcap, fullauxiliarytarget500MiB. No53k-square distance allocation.'
write(p,state)
report=BASE/'RESUMEN_RESULTADOS.md';text=report.read_text(encoding='utf-8-sig')
old='El siguiente estudio examina el mecanismo vial de UNITE y primero comprueba que no use ruta ni longitud reales en una consulta OD.'
new='El estudio vial pasó la prueba sintética de terminación e independencia entre consultas con componentes temporales reales de UNITE, sin entrenar todavía un modelo Porto. El emparejamiento de rutas completas falló su umbral de calidad; una nueva supervisión parcial conserva el 99,14 % del tiempo en intervalos fiables y mantiene todas las etiquetas originales. Se ejecuta un diagnóstico cronológico solo en Train para medir el aporte de conocer la ruta real; su MAE no contará como predicción OD.'
assert text.count(old)==1;text=text.replace(old,new)
write(OUT/'registro_instrumentos_viales.json',dict(completed=state['completed_new_instruments'],active_trial=state['current_trial'],Test='sealed',not_primary_scores=True))
report.write_text(text,encoding='utf-8')
print(json.dumps(dict(recorded=True,best=state['retained_best']['validation'],Test='sealed')))
