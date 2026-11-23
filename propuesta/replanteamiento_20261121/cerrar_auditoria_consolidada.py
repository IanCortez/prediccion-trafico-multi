from base_replanteo import *
p=OUT/'consolidar_auditoria_resultados.py';s=p.read_text(encoding='utf-8-sig')
s=s.replace("ax.invert_yaxis();ax.set_xlim(0,165);", "ax.invert_yaxis();ax.set_ylim(len(chosen)+.5,-.5);ax.set_xlim(0,165);")
p.write_text(s,encoding='utf-8')
c=read(OUT/'CONTINUIDAD_OPERATIVA.json');c['last_goal_turn_classification']=dict(previous_turn='progress',evidence=['Full native result confirmed; temporal history rejected with matched control','Full semantic frozen-head experiment completed, both worse than reference; weights discarded and all predictions retained','CPU auxiliary-probability/gradient gate passed; next matched GPU family queued behind live risk process'])
c['consolidated_full_output_audit']=dict(file='registro_consolidado_auditado.json',outputs=30,archives=13,row_label_timestamp_match=True,reported_metric_max_error=0.0,new_learned_improvements=0,figure='comparacion_resultados_auditados.png',scope='Current replanteamiento family, not a universal impossibility statement')
write(OUT/'CONTINUIDAD_OPERATIVA.json',c)
marker='<!-- auditoria_consolidada_30_salidas -->'
text="""

<!-- auditoria_consolidada_30_salidas -->
Se recalcularon 30 salidas completas a partir de 13 archivos de predicciones. Todas usan exactamente las mismas 127.941 filas, etiquetas y fechas que la referencia; las métricas recalculadas coinciden sin diferencia con los registros. Ninguna salida aprendida de esta familia mejora la referencia. Las copias de la referencia se identifican expresamente y no se cuentan como avances.

[Registro consolidado auditado](registro_consolidado_auditado.json) y [comparación visual](comparacion_resultados_auditados.png). Los ensayos activos no se presentan como resultados. Esta auditoría verifica cifras/cohorte de los archivos de predicciones; no convierte la búsqueda adaptativa en confirmación independiente ni prueba que la meta sea imposible.
"""
p=OUT/'README.md';body=p.read_text(encoding='utf-8-sig')
if marker not in body:p.write_text(body+text,encoding='utf-8')
print('Auditoría consolidada documentada')
