from base_replanteo import *
p=OUT/'consolidar_auditoria_resultados.py';s=p.read_text(encoding='utf-8-sig')
start=s.index("for run,label in [('mediana_directa_s0','Mediana explícita con ajuste conjunto')")
end=s.index('pending=[run',start);block=s[start:end]
old="kind='selected_model',learned=r['learned_update_selected']"
new="kind='selected_model' if r['learned_update_selected'] else 'retained_existing_reference',learned=r['learned_update_selected']"
assert block.count(old)==1;block=block.replace(old,new);s=s[:start]+block+s[end:];p.write_text(s,encoding='utf-8')
c=read(OUT/'conclusion_mediana_directa.json');gate=read(OUT/'piloto_compresion_historial_s0/resultado.json')
state=read(OUT/'CONTINUIDAD_OPERATIVA.json')
state['completed_direct_median']=dict(epochs=6,best_learned=c['best_learned_validation'],selected_epoch=0,new_learned_gain=False,all_metrics_predictions_retained=True,control='Skipped per negative conditional gate.',memory_recovery='Completed from epoch1; original/fixed sources and failure log preserved.')
state['next_historical_context_gate']=gate
state['current_trial']=dict(experiment='datos_historia_individual_s0',stage='CPU2preparation of64historical source IDs; no GPUtraining launched yet.',pid=None,Test='sealed')
state['next_after_result']=['Inspect actual preparar_indices_historia_individual.py process and datos_historia_individual_s0/resultado.json; no blind restart.',
 'Architecture and gate: plan_atencion_historia_individual.json and piloto_compresion_historial_s0/resultado.json. New input is individual pastTrain source values; five-field query unchanged.',
 'Implement shared deterministic28token constructor, Train-only normalization and8slotset attention on frozen parent22; verify Train numerics/exclusion/fallback; freeze and actually train8epochs.',
 'Conditional compression control uses identical64tokens/network/init and prototype-replacedsourceG/T if a real fullValidation gain. Keep all metrics/predictions and eachsub131.06branch checkpoint.',
 'Best130.204821 remains certified/protected; final73.5confirmation unmet. Test closed; do not terminate just because directMAEfailed.']
state['previous_goal_turn_classification']='PROGRESS: historical control terminal/certified,31modelsprotected; negative6epochdirectMAE completed/recovered/audited; Train64sourcecoveragegatepositive24.7percent; next historical-context input bank actually preparing.'
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
for path in [OUT/'README.md',BASE/'RESUMEN_RESULTADOS.md']:
 t=path.read_text(encoding='utf-8-sig')
 old='El ensayo de mediana directa está en ejecución: no tiene todavía un resultado final atribuible.'
 new='La mediana directa terminó seis épocas y no mejoró: mejor época aprendida 131,355547 s; final 132,077286 s. Su selección fue la referencia ya existente, sin ganancia nueva. Se conservan todas sus predicciones; se eliminaron solo intermediarios y duplicados descartados. La siguiente vía examina los 64 viajes históricos individuales: su diagnóstico de cobertura geométrica en Train pasó (24,7 %), sin demostrar todavía mejora del MAE.'
 assert old in t;t=t.replace(old,new).replace('La auditoría recalculó 205 resultados de 25 archivos sin discrepancias.','La auditoría incorpora las seis épocas nuevas; el registro consolidado indica el recuento vigente y las discrepancias.')
 path.write_text(t,encoding='utf-8')
print('Negative direct-median result recorded accurately; identity fallback excluded as a new learned model in the plot.')
