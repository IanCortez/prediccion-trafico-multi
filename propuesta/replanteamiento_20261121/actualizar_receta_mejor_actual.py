from base_replanteo import *
p=BASE/'REPRODUCIR.md';text=p.read_text(encoding='utf-8-sig')
assert '<!-- mejor_actual_atencion_20261122 -->' not in text
intro="""<!-- mejor_actual_atencion_20261122 -->
## Mínimo numérico vigente: 130,158042 s

La atención sobre historias individuales se seleccionó en la época 3. Su diferencia de 0,046779 s frente al padre tiene un intervalo que incluye cero; es un mínimo exploratorio, sin confirmación de mejora fiable. Test sigue cerrado.

El checkpoint nuevo guarda solo 186.848 parámetros de la rama y requiere el padre protegido de la época 27 de historica_fresca_s0. El cargador comprueba ambos pesos, fuentes y normalización; no se debe borrar ese padre. Los datos originales, mapas, perfiles, memoria histórica y bancos requeridos siguen dentro de la carpeta.

Para reproducir las 127.941 predicciones y las consultas crudas sin cambiar la certificación existente:

~~~powershell
Set-Location -LiteralPath 'E:\\UTEC\\Tesis\\TesisIan'
. '.\\auditoria\\entorno.ps1'
& $TaskPython -B '.\\propuesta\\replanteamiento_20261121\\reproducir_atencion_historia.py'
~~~

La comprobación escribe propuesta/replanteamiento_20261121/reproduccion_atencion_historia.json. Para consultas nuevas, inferencia_atencion_historia.py recibe --run atencion_historia_individual_s0, --input y --output; ambos archivos deben estar dentro del proyecto. Cada consulta contiene solo o_lon, o_lat, d_lon, d_lat y timestamp. La salida requiere una ruta nueva y contiene mediana e intervalos exploratorios. No se utiliza la trayectoria, duración, conductor ni longitud reales de la consulta.

La receta anterior reproduce la referencia original, que también permanece protegida.

"""
p.write_text(intro+text,encoding='utf-8')
state=read(OUT/'CONTINUIDAD_OPERATIVA.json')
state['unite_OD_synthetic_instrument']=read(OUT/'instrumento_unite_od_sintetico.json')
state['full_prediction_audit']=dict(outputs=223,archives=27,maximum_reported_metric_error=0,all_rows_labels_timestamps_exact=True)
state['latest_storage_snapshot']=read(OUT/'inventario_conservacion.json')
state['next_after_result'].insert(3,'Synthetic officialforward probe ran:ratio−1targetpayloadinvariant,ratio0lengthsensitive; stoppedrow terminal-time hazard9vs15syntheticunits in mixedbatch. Real SegTimeEstimator has no hard destination-zero mask. Next require absorbing-ODadapter/batchinvariance and inspectTrainstreetsequenceavailability; no claim about an actual trainedUNITEcheckpoint.')
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
print('Current best reproducibility recipe updated; original reference instructions retained.')
