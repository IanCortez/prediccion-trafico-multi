"""Record certified numerical minimum without claiming a meaningful or causal gain."""
from base_replanteo import *
run=OUT/'atencion_historia_individual_s0';r=read(run/'resultado.json');cert=read(run/'certificacion/resultado.json')
assert cert['passed'] and cert['full127941saved_point_predictions_exact'] and cert['reverse_singleton_all_quantiles_exact']
assert r['epochs_run']==8 and sha(r['model_file'])==r['model_sha256']==cert['checkpoint_sha256']
parent=Path(r['config']['parent_model_file']).resolve();assert parent.is_relative_to(OUT.resolve()) and sha(parent)==r['config']['parent_sha256']
for item in r['saved_checkpoints']:assert sha(run/item['file'])==item['sha256']
conclusion=dict(validation=r['validation'],best_epoch=r['best_epoch'],epochs=8,
 paired_vs_parent=r['paired_vs_parent'],meaningful_control_gate_passed=r['control_trigger_met'],
 interpretation='Numerically lower reusedValidation MAE, but0.0468s is below.1s predeclared meaningful threshold and day-bootstrap interval includes0. No reliable or causal accuracy benefit established.',
 control='Not executed: predeclared meaningful-gain gate failed.',saved_threshold_models=len(r['saved_checkpoints']),
 parent_frozen=True,source_mode_mass=cert['uncertainty']['mean_mass_general_public_proto_individual'],
 conclusion='Historical branch is used(~15%mass), so this is not merely a collapsed gate. Geometry coverage improved inTrain, but the time-MAE gain remains weak. Do not expand neighbor/slot counts or tune blend weights on this Validation.',
 all_initial_and_epoch_metrics_predictions_retained=True,
 next_family='Audit/adapt officialUNITE street-segment progress/local-traffic mechanism, with synthetic OD target-poisoning and batch-composition invariance before any real-data training.',
 Test='sealed',goal_confirmation73_5_achieved=False,source_sha256=sha(__file__))
write(OUT/'conclusion_atencion_historia.json',conclusion)
old=read(OUT/'nuevo_mejor_verificado.json')
assert r['validation']['mae_s']<old['validation']['mae_s']
best=dict(id=run.name,epoch=r['best_epoch'],validation=r['validation'],model_file=r['model_file'],model_sha256=r['model_sha256'],
 checkpoint_kind='Branch-only delta; immutable protected parent required.',
 required_parent=dict(id=r['config']['parent_id'],model_file=str(parent),model_sha256=r['config']['parent_sha256']),
 certification=str((run/'certificacion/resultado.json').relative_to(ROOT)),
 scope='Exact127941point replay and raw65fivefields/reverse/singleton. One seed/reusedValidation, not independent confirmation.',
 qualifies_below131_06=True,preserve_original_reference=old['preserve_original_reference'],
 historical_memory_sha256=r['config']['memory_sha256'],all_completed_metrics_predictions_retained=True,
 control='Meaningful-gain gate failed; interval includes0; no causal or robust improvement claim.',
 practical_gain_established=False)
lineagefile=OUT/'modelos_mejores_protegidos.json';lineage=read(lineagefile)
for item in [old,best]:
 if not any(x['model_sha256']==item['model_sha256'] for x in lineage):lineage.append(item)
for item in lineage:
 path=Path(item['model_file']).resolve();assert path.is_relative_to(OUT.resolve()) and sha(path)==item['model_sha256']
write(lineagefile,lineage);write(OUT/'nuevo_mejor_verificado.json',best)
p=OUT/'consolidar_auditoria_resultados.py';s=p.read_text(encoding='utf-8-sig')
marker="for run,label in [('atencion_historia_individual_s0','Atención sobre historias individuales')"
if marker not in s:
 needle="pending=[run for run in ['campo_multiescala_s0','aux_subviajes_s0','atencion_vial_s0','historica_fresca_s0','control_historica_fresca_s0','mediana_directa_s0'] if not (OUT/run/'resultado.json').exists()]"
 block="""for run,label in [('atencion_historia_individual_s0','Atención sobre historias individuales'),('control_compresion_historia_s0','Historial: control con valores resumidos')]:
 source=OUT/run/'resultado.json'
 if source.exists():
  r=read(source);z=load_archive(source.parent/'validacion.npz')
  append(run,label+' — inicial nueva',z['epoch_0'],r['initial_validation'],source,kind='initial_candidate',learned=False)
  append(run,label+' — seleccionada',z['pred'],r['validation'],source,kind='selected_model',learned=r['learned_update_selected'])
  for row in r['all_epoch_validation']:
   epoch=row['epoch'];append(run,label+(' — final fija' if epoch==8 else ' — época '+str(epoch)),z['epoch_'+str(epoch)],row['validation'],source,kind='fixed_final_epoch' if epoch==8 else 'epoch_checkpoint')
pending=[run for run in ['campo_multiescala_s0','aux_subviajes_s0','atencion_vial_s0','historica_fresca_s0','control_historica_fresca_s0','mediana_directa_s0','atencion_historia_individual_s0'] if not (OUT/run/'resultado.json').exists()]"""
 assert s.count(needle)==1;s=s.replace(needle,block);p.write_text(s,encoding='utf-8')
state=read(OUT/'CONTINUIDAD_OPERATIVA.json')
state['goal_status']='ACTIVE; verified numericalValidation minimum130.158042s; weak0.0468s gain, not significant/practical;73.5confirmation unmet,Testsealed.'
state['retained_best']=best
state['completed_individual_history_attention']=conclusion
state['current_trial']=dict(experiment='UNITE_OD_controlflow_and_data_audit',stage='PreviousGPUfamilycomplete/certified. Next dependency/control-flow instruments, no officialscript/Test execution.',pid=None,Test='sealed')
state['next_after_result']=['Current8epochattentioncomplete. Certified130.1580415973s;0.0468s lower than parent,CIcrosses0; gatefailed so no compression control or more k/slot/blend tuning.',
 'Keep40sub131.06model candidates (20historical+11control+9attention), all pointpredictions/metrics, originalbest andTransferTraj. Newbest is branch-only;81c543parent mandatory and protected.',
 'Next: officialUNITElocalcode progress/segmenttraffic adaptation. Run synthetic extractedforward target-poisoning and batch-composition proof first; never run officialdefaults (ODdriver tf_ratio defaults1, GTroute). ODmust−1 and fixed dummytargets.',
 'Portoquery road endpoints must be obtained from5rawfields/publicmap, not from true mapmatchedpath. PrimaryT=(N−1)*15 andcohort unchanged; route/timing source labels onlyTrainauxiliary.',
 'Inspect availableTrain mapmatching arrays/graph data and requiredpackages; no newdataquestions. Build boundedTrain-onlystreetsequence pilot if signal/protocol viable; do not claimUNITEPortopublishedreplication.',
 'Goal73.5confirmation remains active; Testclosed.']
state['previous_goal_turn_classification']='PROGRESS: completed/certified historicalcontrol,negative directMAE, positivecompressioninstrument and actual8epochindividualhistoryattention; protected/audited every completed output. New numerical minimum is weak, so advance different streetprogress family.'
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
header="""<!-- estado_actual_verificado_inicio -->
Mínimo numérico verificado: **MAE 130,158042 s** en los mismos 127.941 viajes de validación. La atención sobre historias individuales terminó ocho épocas y seleccionó la tercera; se reprodujeron todas sus predicciones y las consultas crudas, individuales y en distinto orden.

La diferencia frente al padre congelado de 130,204821 s es solo −0,046779 s, con IC95 % por día [−0,121414; 0,024939]. No pasó el umbral previo de mejora útil de 0,1 s y el intervalo incluye cero. Se conserva el mínimo observado, sin atribuirle una mejora fiable ni ejecutar el control condicional o seguir variando vecinos y pesos.

Se conservan 40 candidatos de estas últimas familias con MAE menor de 131,06 s y las métricas y predicciones de todos los ensayos, incluidos los peores y TransferTraj. El nuevo checkpoint guarda solo la rama de atención y requiere el padre protegido; la inferencia lo carga automáticamente. La mediana directa se descartó (mejor época aprendida: 131,355547 s). El control histórico certificado obtuvo 130,564166 s.

La meta de confirmación menor de 73,5 s sigue pendiente y Test cerrado. La cohorte publicada de TransferTraj es distinta. El siguiente estudio examina el mecanismo vial de UNITE y primero comprueba que no use ruta ni longitud reales en una consulta OD. Evidencia: nuevo_mejor_verificado.json, conclusion_atencion_historia.json, registro_consolidado_auditado.json y atencion_historia_individual_s0/certificacion/resultado.json.
<!-- estado_actual_verificado_fin -->"""
for p in [OUT/'README.md',BASE/'RESUMEN_RESULTADOS.md']:
 text=p.read_text(encoding='utf-8-sig');a=text.index('<!-- estado_actual_verificado_inicio -->');b=text.index('<!-- estado_actual_verificado_fin -->')+len('<!-- estado_actual_verificado_fin -->')
 p.write_text(text[:a]+header+text[b:],encoding='utf-8')
print(json.dumps(dict(numerical_best=r['validation'],practical_gain_established=False,control='skipped predeclared meaningful threshold',saved_qualifying_attention_models=9,goal='ACTIVE')))
