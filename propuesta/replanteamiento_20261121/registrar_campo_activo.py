from base_replanteo import *
# Keep audit coverage current without editing any live model source.
p=OUT/'consolidar_auditoria_resultados.py';s=p.read_text(encoding='utf-8-sig')
old="pending=[run for run in ['historicos_integrados_s0'] if not (OUT/run/'resultado.json').exists()]"
new="""for run,label in [('campo_multiescala_s0','Campo espacial desde cero'),('control_capacidad_campo_s0','Control de capacidad del campo')]:
 source=OUT/run/'resultado.json'
 if source.exists():
  r=read(source);z=load_archive(source.parent/'validacion.npz')
  append(run,label+' — mejor época',z['pred'],r['validation'],source,kind='fresh_selected')
pending=[run for run in ['campo_multiescala_s0'] if not (OUT/run/'resultado.json').exists()]"""
assert old in s;s=s.replace(old,new)
s=s.replace("'selected_risk_weight']","'selected_risk_weight','fresh_selected']")
compile(s,str(p),'exec');p.write_text(s,encoding='utf-8')
c=read(OUT/'CONTINUIDAD_OPERATIVA.json')
c['field_study']=dict(experiment='campo_multiescala_s0',pid=17080,supervisor_pid=32216,session=44092,source_plan='plan_campos_multiescala.json',source_files_frozen=True,initial_GPU_gate_passed=True,protected_reference_all127941_exact=True,embedding_gradients_actually_nonzero=True,parameters=6340040,initialization='from scratch; not another frozen-reference correction',conditional_capacity_control='only if fresh field beats original reference')
c['last_goal_turn_classification']=dict(previous='progress',current='progress and verified wait',evidence=['Full risk study completed and all35outputs audited','Request semantic and out-of-fit teacher student completed; neither beats reference','Historical integration fully evaluated from raw query fields; no gain','Fresh multiscale representation actually training after mathematical/real-GPU gates'])
c['completed_historical_policy']=read(OUT/'conclusion_historicos_integrados.json')
c['next_after_result']=['Inspect actual live field process17080/session44092 and supervisor32216; do not edit model/trainer/data sources while active.',
 'This is fresh joint training with multiscale spatial representation, not a warm residual head. Actual GPU gate and nonzero embedding gradients passed.',
 'Only if field beats original reference will supervisor launch the predeclared comparable-capacity control, one GPU at a time. A Validation target candidate instead takes priority for freeze/reproduction/confirmation.',
 'Preserve every epoch below131.06 and every strict new best; checkpoint/model raw reproduction required before claims; Test remains sealed.',
 'Negative semantic/teacher/history studies are complete and should not be re-tuned with output weights. Original data, strong reference and all TransferTraj metrics/predictions remain protected.',
 'If this fresh representation fails, draw its bounded empirical conclusion and continue a genuinely distinct justified improvement; never mark goal complete from exhausting a batch.']
write(OUT/'CONTINUIDAD_OPERATIVA.json',c)
marker='<!-- campo_multiescala_desde_cero -->'
text="""

<!-- campo_multiescala_desde_cero -->
## Nueva representación espacial y aprendizaje desde cero

Las últimas pruebas terminaron sin mejora. En época4: control131,396816 s; solicitud auxiliar131,486388 s; alumno de maestro probabilístico131,526133 s. La solicitud empeora su control en0,089573 s [0,042085;0,137523] y el alumno en0,129317 s [0,034317;0,223687], intervalos exploratorios por día. Ambos seleccionaron época0 (referencia), sin avance aprendido. La integración de cinco históricos obtuvo131,306891 s; aplicar el bloque semanal correspondiente131,345229 s. Sus controles reprodujeron la referencia. No se seguirán ajustando esas salidas congeladas.

El cambio siguiente afecta la representación y el aprendizaje completo: seis rejillas continuas para origen y seis para destino, con pasos de1.000 a31,25m. La interpolación bilineal combina ocho canales por escala. Se concatenan los dos extremos, su producto y ocho variables de calendario; una proyección de128 unidades modifica los parámetros de los ocho componentes de tiempo/geometría del modelo global. Se aprenden desde cero tanto la torre global como las rejillas. La confianza de cada nodo usa solo frecuencias de coordenadas de Train, n/(n+20), para reducir el efecto de celdas sin apoyo.

```mermaid
flowchart LR
 Q["Origen, destino y salida"] --> F["216 características derivadas"]
 F --> B["Torre global de tiempo y geometría"]
 F --> G["Representaciones locales a seis escalas"]
 G --> H["Origen, destino, interacción y calendario"]
 B --> M["Ocho componentes de duración"]
 H --> M
 M --> P["Distribución y mediana del tiempo"]
 T["Duración y geometría de Train"] -. "Solo pérdidas de entrenamiento" .-> M
```

La prueba CPU pasó interpolación, invariancia de orden, recuperación exacta al anular la rama nueva y gradientes finitos. La prueba GPU real también pasó, reproduciendo todas las predicciones protegidas antes de aprender. Los embeddings recibieron gradientes no nulos en el entrenamiento. El modelo tiene6.340.040 parámetros. Presupuesto fijado: hasta50épocas, parada por paciencia después de20, máximo una hora por brazo. Si supera la referencia se ejecutará un control denso de capacidad comparable con igual inicialización global, datos, objetivo y semillas; si no, se evita ese gasto adicional.

[ODConNet](https://openreview.net/pdf?id=y9tJg8Me65), manuscrito rotulado en revisión paraICLR2026, motiva conceptualmente el uso de varias escalas; no se establece aquí su aceptación final ni se afirma replicarlo. [MDTI](https://arxiv.org/html/2510.05840v1) es un antecedente multimodal de2025 que utiliza trayectorias completas; sus resultados no sustituyen la evaluación OD. La implementación local no recibe la trayectoria real en consulta.

Ver plan_campos_multiescala.json, pipeline_campos_estado.json y campo_multiescala_s0.log. El objetivo de confirmación73,5 s sigue pendiente; Test permanece cerrado.
"""
for p in [OUT/'README.md',BASE/'RESUMEN_RESULTADOS.md']:
 body=p.read_text(encoding='utf-8-sig')
 if marker not in body:p.write_text(body+text,encoding='utf-8')
print('Arquitectura nueva, conclusiones y continuidad documentadas')
