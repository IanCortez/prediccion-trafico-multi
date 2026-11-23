from base_replanteo import *
marker='<!-- cierre_nativo_riesgo_activo_20261122 -->'
text="""

<!-- cierre_nativo_riesgo_activo_20261122 -->
## Resultado completo y ensayo vigente

El generador nativo terminó: 79.960 actualizaciones y evaluación de los mismos 127.941 viajes. Los MAE para pesos 0; 0,05; 0,2; 0,5; 1 fueron 131,256712; 131,339954; 131,947030; 135,526180; 148,755950 s. Ninguna salida mejora la referencia. El peso elegido es cero y reproduce exactamente la referencia; no representa una mejora aprendida. Incluso la diferencia de +0,083241 s del peso 0,05 tiene IC95 % exploratorio por día [0,006151; 0,167739]. Ver conclusion_nativo_completa.json.

El control de historial temporal también terminó. Sobre el estado neuronal congelado, la cabeza con conteos empeoró NLL respecto a la cabeza equivalente sin conteos en 0,015864 [0,014052; 0,017760]. Se descartó esa corrida GPU. La mejora previa frente a conteos más simples no demostraba complementariedad con la red. Se conservan las pruebas y conteos compactos; se retiraron 38,54 MiB de intermediarios terminados o descartados, manteniendo el padre recurrente necesario para el ensayo activo.

Está ejecutándose riesgo_total_s0: ajuste del CRPS del tiempo total de trayectorias completas generadas, con regularización por prefijos reales de Train. La comprobación GPU de gradientes finitos y no nulos pasó. Usa el padre nativo, sin la rama temporal descartada. Las pérdidas parciales no son un MAE de Validación. Ver pipeline_riesgo_estado.json y riesgo_total_s0.log para el avance efectivo.

Un diagnóstico distinto, solo con Train, predijo el tipo de llamada A/B/C desde OD y calendario con exactitud 82,18 % frente a 51,04 % de mayoría. Es una posible etiqueta auxiliar de entrenamiento, no un resultado de duración ni evidencia de información adicional frente a la red fuerte. El tipo observado no podrá entrar en consultas; cualquier propuesta deberá inferirlo y marginalizarlo usando únicamente los cinco campos autorizados.

La meta de confirmación <73,5 s sigue pendiente y Test permanece cerrado. La mejor referencia continúa en 131,256712 s. Se conservará todo candidato completo <131,06 s y cualquier nuevo mejor modelo, junto con métricas, predicciones, fuentes y reproducción. Estos resultados sustituyen los estados pendientes de los apartados históricos anteriores.
"""
for p in [OUT/'README.md',BASE/'RESUMEN_RESULTADOS.md']:
    body=p.read_text(encoding='utf-8-sig')
    if marker not in body:p.write_text(body+text,encoding='utf-8')
write(OUT/'estado_documentado_20261122.json',dict(native_complete=True,temporal_history_gate=False,active='riesgo_total_s0',GPU_gate_passed=True,Test='sealed',goal='active'))
print('Resúmenes actualizados sin modificar fuentes en ejecución')
