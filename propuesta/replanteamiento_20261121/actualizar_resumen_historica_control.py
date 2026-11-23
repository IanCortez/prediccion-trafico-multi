"""Reconcile thesis summaries with the certified comparison and actual next run."""
from base_replanteo import *
c=read(OUT/'conclusion_historica_fresca.json')
assert c['both_full_and_raw_input_certifications_passed']
def fmt(x,n=6):return f'{x:.{n}f}'.replace('.',',')
a=c['treatment']['mae_s'];b=c['control']['mae_s'];delta=c['selected_epoch_paired_treatment_minus_control']['delta_mae_s']
block=f"""<!-- estado_actual_verificado_inicio -->
Resultado vigente verificado: **MAE {fmt(a)} s** en los mismos 127.941 viajes de validación. La mezcla histórica entrenada desde cero terminó 39 épocas y seleccionó la época 27. Se reprodujeron todas sus predicciones y las consultas crudas, individuales y en distinto orden.

El control con la misma inicialización y sin rama histórica terminó 49 épocas: **MAE {fmt(b)} s**, época 40, también certificado. La diferencia entre los modelos seleccionados es {fmt(delta)} s; a la época fija 20 es {fmt(c['fixed_epoch20_paired_treatment_minus_control']['delta_mae_s'])} s. Es un aporte pequeño. Esta ablación cambia la capacidad efectiva; una semilla y validación reutilizada no constituyen confirmación independiente.

Se conservan los 31 candidatos de esta comparación con MAE menor de 131,06 s, además de métricas y predicciones de todas las épocas y de los resultados TransferTraj. La auditoría recalculó 205 resultados de 25 archivos sin discrepancias. El ensayo de mediana directa está en ejecución: no tiene todavía un resultado final atribuible.

La meta de confirmación menor de 73,5 s sigue pendiente y Test cerrado. La cohorte publicada de TransferTraj es distinta; no se declara una comparación equivalente. Evidencia: nuevo_mejor_verificado.json, conclusion_historica_fresca.json, registro_consolidado_auditado.json y mediana_directa_s0.log.
<!-- estado_actual_verificado_fin -->"""
for p in [OUT/'README.md',BASE/'RESUMEN_RESULTADOS.md']:
 text=p.read_text(encoding='utf-8-sig')
 left='<!-- estado_actual_verificado_inicio -->';right='<!-- estado_actual_verificado_fin -->'
 assert text.count(left)==text.count(right)==1
 first=text.index(left);last=text.index(right)+len(right)
 text=text[:first]+block+text[last:]
 text=text.replace('Mejor resultado conservado: 131,2567 s en Validación completa.','Referencia conservada al inicio de esta etapa: 131,2567 s en Validación completa; el resultado vigente está al principio del documento.')
 text=text.replace('| Mejor modelo conservado | 131,256712 | 127.941 |','| Referencia conservada antes de esta reanudación | 131,256712 | 127.941 |')
 p.write_text(text,encoding='utf-8')
state=read(OUT/'CONTINUIDAD_OPERATIVA.json')
state['current_trial']=dict(experiment='mediana_directa_s0',pid=30436,exec_session_id=28304,serial_runner_pid=38708,
 stage='Direct FP32 median;2readout warmup +4joint epochs; existing historical control terminal/certified. One GPU.',
 sources_frozen=True,source_guard='secuencia_mediana_fuentes_congeladas.json',Test='sealed')
state['completed_historical_control']=dict(validation=c['control'],epochs=49,best_epoch=40,full_raw_certification=True,saved_threshold_models=11)
state['completed_historical_comparison']=dict(selected_delta_s=delta,fixed20_delta_s=c['fixed_epoch20_paired_treatment_minus_control']['delta_mae_s'],all31qualifying_models_preserved=True,
 verdict='Small exploratory improvement from historical branch, not a solution to73.5; new objective/readout trial underway.')
state['retained_best']['control']='Completed49epochs and certified; selected40 MAE130.564166s;31totalqualifying historical treatment/control weights protected.'
state['full_prediction_audit']=dict(outputs=205,archives=25,reported_metric_error=0,rows_labels_timestamps_exact=True)
state['latest_storage_snapshot']=read(OUT/'inventario_conservacion.json')
state['queued_next_trial']['stage']='Activated; no longer queued'
state['next_after_result']=['Observe actual directmedianPID30436/session28304 and mediana_directa_s0.log; do not restart solely after observation timeout.',
 'All source_guard files stay unchanged while trial active.',
 'At terminal: inspect all epochs and update primary full-prediction audit. Certify any learned selected best with certificar_mediana_directa.py, then execute matched frozen_control with unchanged trainer only if improvement exists.',
 'Preserve every threshold model, all metrics/predictions, original data, original model and TransferTraj. Continue a justified different alternative if negative; Test stays sealed and goal73.5confirmation remains unmet.']
state['previous_goal_turn_classification']='PROGRESS: completed/certified historical control, exact205outputaudit,31qualifying models protected, Train-only fine-cycle gate rejected, numerically checked explicit-MAE alternative actually launched.'
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
print(json.dumps(dict(best_verified_mae_s=a,control_verified_mae_s=b,protected_threshold_models=31,goal='ACTIVE',new_trial='mediana_directa_s0')))
