from base_replanteo import *
p=OUT/'verificar_conservacion.py';code=p.read_text(encoding='utf-8-sig')
needle="write(OUT/'inventario_conservacion.json',report);print(json.dumps({k:report[k] for k in ['project_GiB','file_count','new_work_MiB','best_model_unchanged','seconds']}))"
replacement="""current=read(OUT/'nuevo_mejor_verificado.json');current_path=Path(current['model_file']).resolve()
assert current_path.is_relative_to(ROOT.resolve()) and sha(current_path)==current['model_sha256']
report.update(current_best_unchanged=True,current_best_model_sha256=current['model_sha256'],current_best_mae_s=current['validation']['mae_s'],
 legacy_best_fields_refer_to_protected_original_reference=True)
write(OUT/'inventario_conservacion.json',report);print(json.dumps({k:report[k] for k in ['project_GiB','file_count','new_work_MiB','best_model_unchanged','current_best_unchanged','current_best_mae_s','seconds']}))"""
assert needle in code;p.write_text(code.replace(needle,replacement),encoding='utf-8')
p=OUT/'consolidar_auditoria_resultados.py';code=p.read_text(encoding='utf-8-sig')
old="new_improvements=[q for q in rows if q['new_learned_improvement']],pending=pending"
new="improved_run_ids=sorted(set(q['run'] for q in rows if q['new_learned_improvement'])),pending=pending"
assert old in code;p.write_text(code.replace(old,new),encoding='utf-8')
policy=read(OUT/'politica_recursos.json')
policy['training_microbatch_applies_to']='Legacy autoregressive generator only; each subsequent model records its own verified batch.'
policy['tabular_mixture_batch']=512;policy['historical_prototype_data_max_MiB']=450;policy['new_route_bank_max_MiB']=500
policy['preservation_qualifying_checkpoints']='All fullMAE<131.06 and strict global new bests; never trade this against cached intermediates.'
write(OUT/'politica_recursos.json',policy)
parent=read(OUT/'nuevo_mejor_verificado.json');prep=read(OUT/'datos_prototipos_historicos_s0/resultado.json')
fix=read(OUT/'datos_prototipos_historicos_s0/calibracion_fp64.json');gate=read(OUT/'verificacion_mezcla_historica_fresca.json')
assert gate['passed'] and fix['rank']==2
plan=dict(hypothesis='Historical route modes from other weeks provide real detours and route-conditioned time priors absent from shortest/freeflow public paths.',
 coverage_gate='Matched8candidates onTrain20k improved oracle geometric error248.99to108.64m; not primaryMAE.',
 initialization_choice='Fresh full neural architecture and initial historical gate.5, so historical inputs are learned from the first epoch instead of only attaching a residual to an already-fitted parent.',
 model='8global unknown-route modes +8public projection modes +6historical prototypes;22joint time/geometry components.',
 training='All neural parameters from scratch, AdamW.001wd.0003,b512;max50epochs,min20,patience12,cap3600seconds;oneGPU.',
 reuse='Only eligibleTrain normalization,bins and physical calibrations; no retained neural weights loaded.',
 controls='If reference beaten, parent_control ablates the historical branch with same initial state and protocol. It is not an equal-effective-capacity claim.',
 reference=parent,history_data_MiB=prep['total_bytes']/1024**2,
 construction='894105eligible quality-checkedTrain source trips; ownfold exclusion for source geometries and times;64neighbors;6weighted geometric medoids;time medians/MAD adjusted to ODscale.',
 calibration='The initial float32polyfit was ill-conditioned from rcond at1million rows; corrected float64Train-only slope.776607,bias1.429573 before any model training. Both versions archived.',
 information=['o_lon','o_lat','d_lon','d_lat','timestamp'],Test='sealed',goal_confirmation_s=73.5)
write(OUT/'plan_historica_fresca.json',plan)
text="""

<!-- historica_fresca_desde_cero -->
La mezcla de proyección quedó verificada en131,125167s(época6), frente a131,248505s de su control(época3). Al comparar la época9 fijada para ambos, la diferencia es-0,273759s. La diferencia seleccionada frente a la referencia original es-0,131545s, intervalo exploratorio por día[-0,194621;-0,074435]. Sigue siendo una mejora pequeña en validación reutilizada, con una sola semilla; no alcanza131,06s ni73,5s. Ambas reproducciones completas y las consultas crudas/independientes pasaron. Los cuatro mejores sucesivos del tratamiento y el control se conservan.

La siguiente propuesta incorpora seis prototipos de rutas reales de otras semanas como modos de geometría y tiempo. En el diagnóstico de20000viajesTrain, dos rutas públicas más seis prototipos redujeron el error geométrico oráculo a108,638779m frente a248,993851m de ocho rutas públicas:140,355057m(56,37%) con igual número de candidatos. La selección no recibe geometría ni duración reales de la consulta. Es evidencia de cobertura, no de rendimientoMAE.

El índice usa894105viajes elegibles deTrain con geometría válida y llegada anterior avalidación. Para cada consulta toma64vecinos por extremos, con pesos espaciales, horarios y tipo de día. Elige seis representantes geométricos y calcula medianas/MAD de tiempos históricos escalados por distanciaOD. Excluye toda la semana-grupo de la consultaTrain. Cambiar masivamente tiempos/geometrías de esa semana en memoria dejó las entradas idénticas. También pasaron reconstrucción desde cinco campos, orden inverso, consulta individual y rechazo de duración adicional. Los prototipos transformados son un prior geométrico latente; no se garantiza que sean rutas navegables.

El banco histórico y su memoria ocupan412,98MiB. Una advertencia de ajuste escalar detectada antes del entrenamiento se corrigió con cálculoTrain en64bits: pendiente0,776607e intercepto1,429573. Se conservan ambas versiones y la explicación; las características y las etiquetas originales permanecen.

Se entrena desde cero la arquitectura completa de22componentes: ocho modos generales, ocho alternativas públicas y seis prototipos históricos, con cabezas compartidas y puertas aprendidas. La puerta histórica empieza con masa0,5para permitir aprender de esa información desde el inicio. Solo se reutilizan estadísticas de entrada y calibraciones ajustadas enTrain; no se cargan pesos de la red anterior. Presupuesto fijado: hasta50épocas, mínimo20, paciencia12, una hora, unaGPU; si mejora131,125167s se ejecutará la ablación sin rama histórica.

El mejor modelo verificado anterior y los resultadosTransferTraj siguen protegidos. Se guardan todas las predicciones/métricas, y los pesos de cada nuevo mejor resultado y candidato menor de131,06s. Test permanece cerrado. Ver plan_historica_fresca.json, historica_fresca_s0.log y CONTINUIDAD_OPERATIVA.json.
"""
for dest in [OUT/'README.md',BASE/'RESUMEN_RESULTADOS.md']:
 old=dest.read_text(encoding='utf-8-sig');assert '<!-- historica_fresca_desde_cero -->' not in old;dest.write_text(old+text,encoding='utf-8')
state=read(OUT/'CONTINUIDAD_OPERATIVA.json')
state['current_trial']=dict(experiment='historica_fresca_s0',pid=34088,exec_session_id=56168,stage='One GPU, full22component architecture trained from scratch; fixed min20/max50epochs,3600secondcap',sources_frozen=True,reference_mae_s=parent['validation']['mae_s'],Test='sealed')
state['next_after_result']=['Verify actual process34088/session56168, not stale files. Never edit active trainer/model/imported sources.','Allow the predeclared20epoch minimum and50epoch/3600secondcap; do not stop because early fresh scores exceed retained reference.','Every checkpoint below131.06 and every strict global new best is retained; every epoch point archive/metric always retained.','Only after actual terminal fresh result beats verified131.125167s, execute parent_control with identical initial state and protocol, one GPU at a time.','Then freeze/certify raw five-field inference, wholeValidation reproduction and uncertainty; Test stays closed until genuinely strong candidate.','Source preprocessing constructor is copied inside the run; no retained neural state loaded. Float64history calibration is the used version.', 'No goal completion from exhausting a batch;73.5confirmation still unmet.']
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
print('Updated current-best guard, fresh-model plan and live process record.')
