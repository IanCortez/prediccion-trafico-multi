"""Onefinalclosure at explicituserrequest, onlyafter experiment and independentreplay."""
from base_replanteo import *
run=OUT/'red_cinetica_Train_s0';r=read(run/'resultado.json');replay=read(run/'reproduccion_independiente.json')
assert r['complete'] and replay['passed']
diagnostic=read(run/'diagnostico_sin_compuerta.json');assert diagnostic['complete']
assert not (run/'conclusion.json').exists(),'Closure is one-shot'
audit=read(OUT/'conservacion_pilotos_recientes.json');assert audit['passed'] and audit['MAE_threshold_excludes_no_results']
assert any(item['id']=='red_cinetica_Train_s0' for item in audit['records'])
preserved=read(OUT/'inventario_conservacion.json')
assert preserved['current_best_unchanged'] and preserved['best_model_unchanged']
assert len(preserved['TransferTraj_predictions_unchanged'])==10 and all(item['unchanged'] for item in preserved['TransferTraj_predictions_unchanged'])
raw=ROOT/'tesis/porto/porto/train.csv';assert sha(raw)=='2891d03ea99ce0fc203daedd434848a1a8baa7a8fb6da10f8d7ff26559efff6d'
select='kinetic' if r['hypothesis_kinetic_passed'] else ('position' if r['hypothesis_position_passed'] else None)
primary=None
if select is not None:
 primary=read(OUT/'validacion_ultima_red_cinetica_s0/resultado.json')
 assert primary['complete'] and primary['selected_arm']==select and read(OUT/'validacion_ultima_red_cinetica_s0/reproduccion_independiente.json')['passed']
reference=read(OUT/'referencia_publica_Train_red_vial.json')
findings=[
 'Ambas mezclas empeoran la referencia en las mismas10000consultas: posicion+2.985s(IC95%2.089..3.956) y movimiento+2.3625s(IC95%1.446..3.375).',
 'Movimiento frente a posicion cambia el MAE-.6225s, IC95%[-1.510,.241]; no acredita el beneficio de2s registrado.',
 'La red aislada obtiene280.248s(posicion) y268.386s(movimiento), con sesgos medios de sobreestimacion216.987s y203.136s.',
 'Hay soporte de red en9924de10000consultas, pero esa cobertura no se traduce en precision. Las76sin soporte conservan exactamente la referencia.',
 'Las compuertas asignan pesos medios de2.285% y2.075% a las redes y medianas practicamente nulas; aun asi empeoran. Esto indica que la mezcla rechaza casi siempre un componente poco predictivo.',
 'Los calculos probabilisticos y la exclusion de datos pasan sus pruebas, mientras la hipotesis predictiva falla. No se confunde correccion del codigo con validez empirica de la propuesta.',
 'Esta arquitectura no queda respaldada como mejora final para la tesis. Se conserva el mejor modelo primario anterior y se documenta el resultado negativo. No se abren nuevas tandas por indicacion del usuario.'
]

dependencies=[OUT/'conteos_cineticos_prefijo_Train_s0/conteos_fuentes.npz',
 OUT/'indices_red_cinetica_Train_s0/consultas_predicciones.npz',Path(reference['model_file']),
 run/'position/epoca_80.pt',run/'kinetic/epoca_80.pt']
conserved=[dict(file=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in dependencies]
counts=read(OUT/'conteos_cineticos_prefijo_Train_s0/resultado.json')
features=read(OUT/'indices_red_cinetica_Train_s0/resultado.json')
limitations=r['limitations']+[
 'La transicion depende de calle, posicion, clase de movimiento y hora, pero no aprende una politica especifica de destino; condicionar la posicion final no recupera todas las decisiones de ruta.',
 'La coincidencia de mapa de las fuentes no equivale a una anotacion humana de la calle correcta.',
 'La exclusion semanal de fuentes y las pruebas numericas acreditan integridad del procedimiento; no acreditan una mejora predictiva.',
 'TransferTraj publica73.5s en otra cohorte procesada de Porto(323481viajes), que no esta disponible aqui. Esa cifra es una meta externa y no una replica comparable directa.',
]
conclusion=dict(complete=True,user_instruction='Termina esta ultima propuesta, y ahi cerramos',
 proposal='Red vial con estados de posicion/movimiento y reloj original15s; mezcla probabilistica con referencia publica congelada.',
 metrics=r['metrics'],comparisons=r['comparisons'],critical_findings=findings,hypothesis_position_passed=r['hypothesis_position_passed'],
 hypothesis_kinetic_passed=r['hypothesis_kinetic_passed'],selected_for_primary=select,primary_result=primary,
 independent_complete_raw5_replay=replay,primary_best_s=preserved['current_best_mae_s'],
 confirmation_target_s=73.5,confirmation_target_achieved=False,
 retained_results_regardless131_06=True,retained_inference_andreplay_dependencies=conserved,
 temporary_source_shards_retired=read(OUT/'conteos_cineticos_prefijo_Train_s0/retiro_shards_verificados.json')['bytes'],
 temporary_prediction_shards_retired=read(OUT/'indices_red_cinetica_Train_s0/retiro_shards_predicciones.json'),
 original_CSV_sha256=sha(raw),original_data_unchanged=True,TransferTraj_archives_unchanged=10,
 project_GiB=preserved['project_GiB'],sourcecounts_MiB=counts['artifact_MiB'],predictionfeatures_MiB=features['artifact_MiB'],
 limitations=limitations,postfixed_rawgraph_diagnostic=diagnostic,Test='SEALED',further_experiments='CLOSED_BY_EXPLICIT_USER_REQUEST_AFTER_THIS_PROPOSAL')
# Thisguard prevents claiming73.5 without an actualindependentconfirmation artifact.
if primary is not None and primary.get('confirmation_goal_achieved'):
 raise AssertionError('A confirmedcandidate requires an explicitconfirmationlinkedclosure, not this no-confirmation report')
write(run/'conclusion.json',conclusion)
report=[]
report.append('CIERRE DE LA ULTIMA PROPUESTA DE LA TESIS DE IAN')
report.append('Trabajo realizado exclusivamente en E:/UTEC/Tesis/TesisIan, con su entorno Conda local.')
report.append('')
report.append('Resultado: propuesta completada, reproducida y cerrada por indicacion del usuario. La meta de MAE menor de73,5segundos no esta acreditada.')
report.append('')
report.append('EXPERIMENTO FINAL — mismas10000consultas de reserva dentro deTrain, reutilizadas exploratoriamente; no son Test.')
for arm,label in [('reference','Referencia publica congelada'),('position','Red vial con posicion'),('kinetic','Red vial con posicion y movimiento')]:
 m=r['metrics'][arm];report.append(f"{label}: MAE {m['mae_s']:.6f}s; RMSE {m['rmse_s']:.6f}s; n={m['n']}.")
for label,pair in r['comparisons'].items():
 lo,hi=pair['ci95_s'];report.append(f"{label}: cambio de MAE {pair['delta_mae_s']:.6f}s; IC95% por dia [{lo:.6f}, {hi:.6f}]s. Un valor negativo favorece al primer modelo.")
report.append('Criterio previo: posicion debe mejorar al menos5s; movimiento debe mejorar5s frente a referencia y2s frente a posicion, con intervalos por dia por debajo de cero.')
report.append('Criterio de posicion cumplido: '+str(r['hypothesis_position_passed'])+'. Criterio de movimiento cumplido: '+str(r['hypothesis_kinetic_passed'])+'.')
if primary is None:report.append('No se amplio a la validacion primaria porque no supero los criterios fijados antes de conocer los errores.')
else:report.append('Validacion primaria: '+json.dumps(primary,ensure_ascii=False))
report.append('')
report.append(f"Mejor MAE primario protegido: {preserved['current_best_mae_s']:.6f}s sobre127941consultas. Test permanece cerrado. Un resultado enTrain no sustituye la confirmacion independiente.")
report.append('')
report.append('QUE SE IMPLEMENTO Y COMPROBO')
report.append('El estado representa calle dirigida, posicion y tres clases de movimiento. Cada paso conserva los15segundos de muestreo original. Se suman las rutas y duraciones posibles bajo el modelo en vez de seleccionar un recorrido codicioso.')
report.append('Las consultas contienen solo origen, destino y hora. No reciben su numero de puntos, duracion real, velocidad real ni calle real. Se mantienen todas las consultas y etiquetas; cuando la red carece de soporte se devuelve exactamente la referencia.')
report.append('Las fuentes se limitan a951718viajes que terminan antes del corte temporal deTrain. Para ajustar las compuertas se excluye toda la particion semanal de la consulta antes de construir conteos, normalizadores y soportes.')
report.append('Ambas compuertas reciben los mismos206atributos publicos, la misma inicializacion y12560actualizaciones; no hay parada temprana segun los errores de reserva.')
report.append('La reproduccion independiente recalculo las10000distribuciones de red desde los cinco campos de consulta, ambas compuertas, las predicciones y sus estadisticas. Las diferencias numericas de GPU se documentan con tolerancias; no se declaran igualdad exacta cuando no corresponde.')
report.append('')
report.append('INTERPRETACION CRITICA')
for finding in findings:report.append('- '+finding)
for limitation in limitations:report.append('- '+limitation)
for kind,item in diagnostic['diagnostics'].items():
 report.append(f"Diagnostico de {kind}: red sin compuerta MAE {item['rawgraph_with_reference_fallback_metrics']['mae_s']:.6f}s; peso medio de la red en la mezcla {item['gate_weight_mean']:.6f}; consultas sin soporte {item['unsupported_fallback_count']}. Todas las10000consultas se conservan.")
report.append('Un resultado negativo se conserva como evidencia y no justifica ampliar rejillas de parametros ni asegurar que una arquitectura mas compleja alcanzara la meta.')
report.append('')
report.append('CONSERVACION Y ESPACIO')
report.append('Se conservaron resultados favorables y desfavorables, incluidos MAE superiores a131,06s: metricas, predicciones, probabilidades, configuraciones, codigo y conclusiones.')
report.append('Los datos originales, la linea de mejores modelos y los10archivos de predicciones de TransferTraj permanecen verificados. Se retiraron las copias temporales de conteos y predicciones despues de verificar sus archivos finales.')
report.append(f"Conteos de inferencia: {counts['artifact_MiB']:.2f}MiB. Datos compactos para reproducir el experimento: {features['artifact_MiB']:.2f}MiB. Proyecto completo medido: {preserved['project_GiB']:.3f}GiB.")
report.append('')
report.append('ARCHIVOS PRINCIPALES')
report.append(str(run/'resultado.json'));report.append(str(run/'predicciones_held_Train.npz'));report.append(str(run/'reproduccion_independiente.json'));report.append(str(OUT/'protocolo_red_cinetica_Train.json'))
report.append(str(OUT/'conservacion_pilotos_recientes.json'));report.append(str(OUT/'inventario_conservacion.json'))
report.append('')
report.append('REPRODUCCION')
report.append("Desde E:/UTEC/Tesis/TesisIan, cargar auditoria/entorno.ps1 y ejecutar con $TaskPython -B propuesta/replanteamiento_20261121/reproducir_red_cinetica_Train.py. Reproduce resultados ya conservados sin reentrenar ni abrir Test.")
report.append('Para nuevas consultas de cinco campos, inferir_red_cinetica_raw5.py acepta --queries archivoJSON --output archivoNPZ --arm position|kinetic, siempre dentro del proyecto. Estos modelos corresponden al pilotoTrain y no reemplazan automaticamente al mejor modelo primario.')
report.append('')
report.append('La busqueda de nuevas alternativas queda cerrada por indicacion del usuario.')
summary=OUT/'RESUMEN_FINAL_ULTIMA_PROPUESTA.txt'
import re
text_report=re.sub(r'(?<=\d)(?=[A-Za-zÁÉÍÓÚáéíóúñÑ])',' ','\n'.join(report))
for before,after in {'posicion':'posición','publica':'pública','publicos':'públicos','duracion':'duración','indicacion':'indicación','validacion':'validación','informacion':'información','configuracion':'configuración','ejecucion':'ejecución','compuertas':'compuertas','codicioso':'codicioso','politica':'política','especifica':'específica','anotacion':'anotación','numericas':'numéricas','ecuaciones':'ecuaciones','prediccion':'predicción','predicciones':'predicciones','interpretacion':'interpretación','conservacion':'conservación','reproduccion':'reproducción','busqueda':'búsqueda','mas':'más','estan':'están','esta':'está','conservo':'conservó','amplio':'amplió','conservacion':'conservación','codigo':'código','metricas':'métricas'}.items():
 text_report=re.sub(r'\b'+before+r'\b',after,text_report)
summary.write_text(text_report+'\n',encoding='utf-8')
state=read(OUT/'CONTINUIDAD_OPERATIVA.json');state['status']='CLOSED_AFTER_FINALPROPOSAL_BY_USER';state['last_closed_timepilot_id']='red_cinetica_Train_s0';state['last_closed_timepilot']=conclusion;state['next']='NONE. Userexplicitlyrequestedfinishthisproposal thenclose. Do notstartanotherfamily.';state['Test']='SEALED'
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
for name in ['arquitectura_activa.json','plan_red_cinetica_15s_Train.json']:
 path=OUT/name
 doc=read(path);doc['status']='COMPLETE_AND_CLOSED_BY_USER';doc['final_experiment']='red_cinetica_Train_s0/resultado.json';doc['final_conclusion']='red_cinetica_Train_s0/conclusion.json';doc['independent_replay']='red_cinetica_Train_s0/reproduccion_independiente.json';doc['next_actions']=[];doc['new_families_authorized']=False;write(path,doc)

closure=read(OUT/'CIERRE_ULTIMA_PROPUESTA.json');closure.update(status='COMPLETE_AND_CLOSED_BY_USER',summary=str(summary),conclusion=str(run/'conclusion.json'),target_achieved=False,Test='SEALED');write(OUT/'CIERRE_ULTIMA_PROPUESTA.json',closure)
print(json.dumps(dict(complete=True,summary=str(summary),metrics=r['metrics'],primary_best_s=preserved['current_best_mae_s'],target_achieved=False,project_GiB=preserved['project_GiB'],further_experiments='CLOSED_BY_USER'),ensure_ascii=False),flush=True)
