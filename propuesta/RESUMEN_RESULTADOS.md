# Resumen de resultados conservados

La meta MAE de confirmación <73,5 s todavía no se ha alcanzado. Test permanece cerrado y el trabajo continúa; el estado vigente está en el apartado de replanteamiento continuo.

| Referencia | Validación: MAE (s) | Viajes |
|---|---:|---:|
| Original corregido | 140,822594 | 127.941 |
| Mejor modelo conservado | 131,256712 | 127.941 |

La mejora frente al original corregido fue de 9,565882 s, aproximadamente 6,8 %. El mejor modelo tiene 4.280.768 parámetros y 216 características derivadas de origen, destino y salida. El checkpoint se seleccionó en la época 53; su evaluación conservada usa float32 y lote canónico 2.048. Se conservaron pesos, fuentes congeladas, selección, historial, predicciones y recursos para entrenamiento e inferencia.

En el ciclo histórico previo a la reanudación, las iteraciones posteriores no justificaron aumentar el costo de aquellos mecanismos. En la reserva interna reutilizada de 4.096 viajes, ocho bancos obtuvieron 130,555825 s frente a 131,286107 s del control: mejora de 0,730283 s, inferior al mínimo fijado y con intervalo que incluía cero. Es una muestra distinta de la validación primaria. El modelo pesado TabFM y sus correcciones no superaron el control en sus pruebas internas. Las últimas anotaciones/cabezas se interrumpieron: no se les atribuye un resultado predictivo.

Las búsquedas fueron adaptativas. Los intervalos internos no corrigen toda esa selección. Ni las pruebas privilegiadas con trayectoria observada ni las corridas incompletas cuentan para la meta OD.

[TransferTraj: métricas y predicciones conservadas](comparaciones_transfertraj/README.md). El archivo identifica siete modelos OD de esa familia, dos controles con representaciones y un intento incompleto. La cohorte publicada de TransferTraj no estaba disponible: estas cifras corresponden al protocolo local, no al 1,225 min reportado en su benchmark.

Los resultados tabulados completos están en `resultados/resultados_experimentos.csv` y su JSON. Se distingue la elegibilidad según la auditoría original y los diagnósticos privilegiados. `resultados/resultados_internos.json` separa los estudios internos. El archivo `resultados/historial_evidencia.zip` conserva, sin duplicados de contenido, código y evidencia ligera; no conserva los tensores ni modelos descartados.

[Cómo reproducir el modelo](REPRODUCIR.md). Las verificaciones antes/después de la limpieza y el inventario están en `../auditoria/limpieza/`.

La limpieza redujo el proyecto de 74.49 a 13.85 GiB (reducción aproximada de 60.64 GiB, 81.4 %). Los 407 archivos protegidos conservaron sus huellas. La verificación posterior reprodujo exactamente las 127.941 predicciones y las consultas crudas. Detalle: `../auditoria/limpieza/resumen_limpieza.json`.

<!-- operativo_20261121 -->
## Reanudacion operativa del 21 de noviembre de 2026

Se ejecutaron siete entrenamientos o ajustes y una simulacion causal con cuatro variantes. La meta <73,5 s sigue sin alcanzarse. El modelo conjunto con rutas globales entrenado desde cero termino tras 63 epocas; su mejor estado obtuvo 131,399512 s en float32, frente a 131,256712 s del conservado. La diferencia pareada fue +0,142800 s, IC95% por dia [-0,083745; 0,355441]: no justifico sustituir el modelo.

La transferencia regional selectiva no supero el control. El ajuste directo al MAE y los adaptadores de distribucion tampoco mejoraron el original. El estado causal local alcanzo 130,418913 s bajo un protocolo online distinto, con viajes ya completados y demora de publicacion de 60 s; no certifica la meta del benchmark estatico.

Una cota diagnostica de 123,835791 s descarta alcanzar la meta mediante mezclas convexas de los dos modelos fijos evaluados, incluso con seleccion privilegiada. No se generaliza a otros modelos ni al error irreducible.

Se conservaron historias, configuraciones, codigo y predicciones distintas; se eliminaron 106.626.152 bytes de pesos, caches de rutas y duplicados descartados. El modelo original y los diez archivos de predicciones TransferTraj conservan sus huellas. La verificacion final reprodujo exactamente todas las predicciones del modelo principal y las consultas OD crudas. Al cerrar aquel ciclo no había experimentos activos; la reanudación vigente se describe abajo. Test permanece cerrado.

[Resultados y limites del ciclo](desarrollo_20261121/README.md). [Registro estructurado](desarrollo_20261121/resumen_resultados.json).

<!-- replanteamiento_activo_20261121 -->
## Replanteamiento continuo activo

El objetivo de confirmación <73,5 s permanece activo. Test sigue cerrado. La mejor referencia conservada continúa en 131,256712 s. Por indicación del usuario, los nuevos candidatos con MAE de Validación completa <131,06 s conservarán pesos, métricas, predicciones y reproducción; se protegen también el modelo anterior y todos los resultados TransferTraj.

| Mecanismo evaluado | MAE en los mismos 127.941 viajes |
|---|---:|
| Descomposición movimiento/bajo desplazamiento supervisado | 131,627313 s |
| Control con la misma descomposición, sin supervisión del conteo | 131,417567 s |
| Memoria de etiquetas con métrica entrenable, distribución | 137,431590 s |
| Memoria de etiquetas con métrica entrenable, media/MSE | 144,303155 s |
| Proceso espacial semi-Markov sin dirección | 138,223439 s |
| Memoria de residuos fuera de bloque | 136,305803 s |
| Proceso espacial con dirección | 137,918025 s |
| Dirección y prior de conteos | 137,538866 s |
| Permanencias por fases y referencia fuerte | 132,700776 s |

No mejoran la referencia. La supervisión de bajo desplazamiento empeora 0,2097 s respecto a su control en esta configuración (IC95 % exploratorio por día: 0,0442 a 0,3703 s); no se generaliza ese resultado a todas las formas de modelar paradas.

La memoria de residuos tampoco mejoró su base (136,186084 s). Ambas variantes terminaron sin mejorar la referencia. La corrección física con permanencias por fases terminó 6.000 actualizaciones, con 96.000 consultas para el tiempo total y 384.000 estados locales. Obtuvo 132,700776 s, peor que la referencia en 1,444064 s (IC95 % exploratorio por día: 1,236382 a 1,665372 s). La rama espacial obtuvo 141,159167 s: mejoró dentro de su familia, pero sigue por encima de la referencia. Se reprodujeron exactamente las 127.941 predicciones de referencia antes del ajuste; las distribuciones se mantienen solo en RAM. La calibración única declarada de antemano terminó: el peso elegido de 0,02 obtuvo 131,257416 s; ninguno de los ocho pesos mejoró la referencia. No se repetirán búsquedas de pesos para esa rama congelada.

Se conservan predicciones y fuentes de ensayos descartados, retirando sus pesos. No se generan bancos CDF por viaje en disco. [Propuesta, conclusiones y ejecución](replanteamiento_20261121/README.md); [política de conservación](replanteamiento_20261121/politica_conservacion.json).

<!-- transicion_prefijo_nativo -->
El piloto de contexto compartido por viaje no justificó un entrenamiento completo: la diferencia de NLL frente a un contexto independiente por tramo fue +0,000183 por observación (IC95 % por día: −0,000264 a +0,000635), en semanas de Train reservadas solo para esos nuevos parámetros escalares. No es una prueba de independencia general. La siguiente alternativa es un generador recurrente de zona, permanencia y finalización a intervalos nativos de 15 s. En consulta usará únicamente su propia trayectoria generada y un horizonte fijo de 120 pasos, sin conocer la longitud real. Pasó pruebas de causalidad, gradientes, conservación de probabilidades y reconstrucción exacta de 1.000 viajes de Train. Todavía no tiene un MAE. El entrenamiento recurrente terminó diez épocas y se encuentra en evaluación completa.

El estado vigente de las ejecuciones está en pipeline_prefijo_estado.json y CONTINUIDAD_OPERATIVA.json, dentro de replanteamiento_20261121. Las métricas parciales o de selección no sustituyen los 127.941 viajes de Validación.

<!-- calibracion_completa_prefijo_activo -->
La calibración terminó con MAE 131,257416 s para el peso elegido, frente a 131,256712 s de referencia. No hay candidato nuevo <131,06 s. El generador de prefijos terminó diez épocas; la evaluación completa sigue activa. Pasó gradientes finitos con datos reales de Train en GPU y reprodujo exactamente las 127.941 predicciones de referencia. Todavía no hay MAE completo del generador.

El inventario tras retirar la reanudación terminada fue 13,9553 GiB. El mejor modelo y los diez archivos de predicciones TransferTraj conservaron sus huellas. Ver inventario_conservacion.json; el tamaño puede crecer temporalmente con el punto de reanudación del entrenamiento activo.

<!-- diagnostico_historia_Train -->
El diagnóstico de historial temporal encontró una mejora local en 9.198.270 decisiones reservadas de Train: NLL 0,558328 → 0,541826, con intervalo por día de la diferencia por debajo de cero. Todavía no demuestra mejora del MAE del viaje ni información adicional frente al modelo neuronal. Antes de otra corrida GPU se compararán cabezas equivalentes con/sin historial y el modelo neuronal original, sobre el padre congelado y sin usar GPS de Validación o Test. El estado vigente sigue en los registros de ejecución.

<!-- control_riesgo_total_preparado -->
Quedó preparado un objetivo distinto sobre el tiempo total de trayectorias generadas. Su estimador pasó enumeración exacta, un muestreador sintético y gradientes con cuatro consultas reales de Train, sin actualizar pesos. El ensayo requiere después una prueba GPU y validación completa; estos instrumentos no demuestran una mejora de MAE. La ejecución seguirá secuencialmente tras los controles en curso y conservará cualquier candidato <131,06 s, además del mejor modelo en todo caso.
