# Replanteamiento operativo activo

Objetivo: MAE de confirmación <73,5 s con la misma cohorte y consulta OD. Test sigue cerrado. Mejor resultado conservado: 131,2567 s en Validación completa.

## Resultados de esta etapa

- base_oof_estatica_s0: {"validation": {"n": 127941, "mae_s": 136.18608419982218, "rmse_s": 217.50664256249215, "mdae_s": 83.49401562688035}, "OOF": {"n": 1023485, "mae_s": 131.475149159145, "rmse_s": 212.84137741731294, "mdae_s": 80.09576416015625}, "in_sample": {"n": 1023485, "mae_s": 126.32435574647819, "rmse_s": 207.88454994783876, "mdae_s": 75.50473022460938}, "seconds": 1764.1730108261108}
- cabeza_fases_cpu: {"seconds": 16.26175308227539}
- datos_semi_markov: {"seconds": 123.99043798446655}
- descompuesto_aux0_s0: {"validation": {"n": 127941, "mae_s": 131.41756747250685, "rmse_s": 212.51456124532928, "mdae_s": 75.0}, "best_epoch": 48, "epochs_run": 68, "seconds": 3320.6846771240234}
- descompuesto_aux1_s0: {"validation": {"n": 127941, "mae_s": 131.62731258939667, "rmse_s": 212.15485612842585, "mdae_s": 75.0}, "best_epoch": 48, "epochs_run": 68, "seconds": 2367.459737062454}
- espacial_direccional_s0: {"validation": {"n": 127941, "mae_s": 137.91802471451686, "rmse_s": 217.45644950373136, "mdae_s": 90.0}, "od_validation": {"n": 127941, "mae_s": 138.0847421858513, "rmse_s": 218.18612603316336, "mdae_s": 90.0}, "graph_validation": {"n": 127941, "mae_s": 152.22395479166178, "rmse_s": 232.5844808193083, "mdae_s": 105.0}, "seconds": 1851.4751737117767}
- espacial_prior_cf_s0: {"validation": {"n": 127941, "mae_s": 137.53886557084905, "rmse_s": 216.9405669865068, "mdae_s": 90.0}, "od_validation": {"n": 127941, "mae_s": 138.09470771683823, "rmse_s": 218.09671610099505, "mdae_s": 90.0}, "graph_validation": {"n": 127941, "mae_s": 148.20280441766127, "rmse_s": 228.17326791366307, "mdae_s": 90.0}}
- espacial_semimarkov_s0_v2: {"validation": {"n": 127941, "mae_s": 138.22343892888128, "rmse_s": 220.15717408778448, "mdae_s": 90.0}, "od_validation": {"n": 127941, "mae_s": 138.33962529603488, "rmse_s": 220.7329296653693, "mdae_s": 90.0}, "graph_validation": {"n": 127941, "mae_s": 154.22175065069055, "rmse_s": 234.46667919909925, "mdae_s": 105.0}, "seconds": 1369.4464671611786}
- memoria_cdf_s0: {"validation": {"n": 127941, "mae_s": 137.4315895608132, "rmse_s": 219.76714831713943, "mdae_s": 90.0}, "best_epoch": 19, "epochs_run": 31, "seconds": 1326.5635328292847}
- memoria_mse_s0: {"validation": {"n": 127941, "mae_s": 144.3031549620929, "rmse_s": 215.5584713506147, "mdae_s": 98.6387939453125}, "best_epoch": 10, "epochs_run": 22, "seconds": 834.9217772483826}
- memoria_residual_oof_s0: {"validation": {"n": 127941, "mae_s": 136.30580321953553, "rmse_s": 216.76987131695302, "mdae_s": 84.3558398865672}, "best_epoch": 22, "epochs_run": 34, "seconds": 1050.101538658142}

## Conclusiones y siguientes pruebas

La memoria de duraciones sin base no mejora el modelo conservado. La variante de residuos fuera de bloque terminó en 136,3058 s, frente a 136,1861 s de su base; tampoco mejora en MAE. No se vuelve a ampliar esta variante sin una hipótesis adicional. Se prueba memoria de residuos calculados fuera de cada bloque; consultas y candidatos de entrenamiento pertenecen al mismo bloque excluido para impedir que la base transmita la etiqueta de consulta.
El proceso espacial sin dirección obtuvo 138,22 s en toda Validación. Su cabeza OD obtuvo 138,34 s y la rama espacial aislada con respaldo en consultas sin soporte, 154,22 s. Estas son comparaciones entre cabezas del mismo modelo, no una ablación con entrenamiento separado. La integración actual apenas aporta respecto a su propia cabeza directa y queda por encima de la referencia TabM.
En Train con semanas reservadas, añadir dirección entrante reduce NLL de siguiente zona de 0,77217 a 0,64690 y sube acierto de 72,75 % a 76,37 %. Esto justifica ampliar el estado, pero NO demuestra beneficio en MAE de duración. La nueva inferencia conserva dirección hipotética de cada transición; la dirección real de la consulta nunca se entrega. Se ejecuta también una corrección neuronal sobre un prior de conteos locales, excluyendo de esos conteos todo el bloque de la consulta Train; este prior no excluye el ajuste neuronal ni el grafo permitido entrenado con Train.
La descomposición liga tiempos de movimiento y bajo desplazamiento mediante convolución; el control sin esa supervisión terminó en 131,4176 s y el supervisado en 131,6273 s. El incremento de error supervisado es 0,2097 s, IC95 % exploratorio por día [0,0442; 0,3703]. Bajo desplazamiento es una aproximación GPS, no una etiqueta humana de parada.
Un selector con acceso a la verdad que eligiera entre las cinco salidas anteriores tendría 100,44 s. Es un diagnóstico privilegiado e inutilizable como predictor; limita solo la selección entre esas cinco salidas, no nuevas funciones ni otras arquitecturas.

## Ejecución vigente

El entrenamiento con permanencias por fases terminó en 132,700776 s de MAE en los 127.941 viajes; la rama espacial obtuvo 141,159167 s. La diferencia pareada del modelo integrado frente a la referencia fue +1,444064 s, IC95 % exploratorio por día [1,236382; 1,665372]. No mejora al modelo conservado.

La calibración única declarada en protocolo evalúa ocho pesos sin repetir la generación. Elegir el peso cero equivale al modelo anterior y no constituye un nuevo resultado aprendido. Las predicciones distintas y todas sus métricas se conservan.

El piloto CPU no respalda compartir un único efecto de demora por todo el viaje frente a un efecto independiente por tramo: diferencia de NLL +0,000183, IC95 % [−0,000264; +0,000635]. Por ello no se inicia un entrenamiento completo de esa hipótesis. Las pruebas sintéticas de ese instrumento sí pasaron; un instrumento correcto puede dar un resultado negativo en los datos.

La siguiente propuesta es un generador recurrente de celdas a la frecuencia nativa de 15 s. Aprende a desplazarse, permanecer y terminar con prefijos reales de Train. Al consultar, la historia y el tiempo transcurrido son los que el propio generador simula; nunca se le entrega la longitud o la trayectoria real. Se acumula probabilísticamente la finalización de 128 partículas en cada paso de un horizonte fijo de 120 pasos. La semilla depende solo de OD/salida, partícula y paso.

Las pruebas CPU pasaron: equivalencia de procesamiento secuencial y por lote, causalidad ante cambios del futuro, independencia del relleno, gradientes finitos, absorción conocida y conservación de masa. En 1.000 viajes reales de Train, duración y extremos coinciden exactamente. El estado recurrente y la estimación por partículas añaden riesgo de desajuste entre entrenamiento y generación y variación Monte Carlo; no se presupone una mejora del MAE. UNITE (KDD 2026) es un antecedente de generación recurrente; este desarrollo no se presenta como réplica exacta ni como novedad por sí solo.

El supervisor continuar_a_prefijo.py espera mediante identificadores originales del sistema operativo a que terminen la calibración y su supervisor. Así evita confundir un PID posteriormente reutilizado con el proceso que terminó. Solo inicia un trabajo GPU.

Consultar pipeline_prefijo_estado.json, CONTINUIDAD_OPERATIVA.json, transicion_a_prefijos.json y los registros de cada ejecución. Los identificadores escritos en apartados históricos no indican procesos vivos.

El objetivo de confirmación <73,5 s permanece activo. Test continúa cerrado. Se conserva cada candidato con MAE completo <131,06 s, junto con el mejor modelo en todo caso, los datos originales y las métricas y predicciones TransferTraj. Un ajuste que termina o falla no cierra la investigación.

<!-- calibracion_completa_prefijo_activo -->
La calibración terminó con MAE 131,257416 s para el peso elegido, frente a 131,256712 s de referencia. No hay candidato nuevo <131,06 s. El generador de prefijos está entrenando; pasó gradientes finitos con datos reales de Train en GPU y volvió a reproducir exactamente las 127.941 predicciones de referencia. Todavía no hay MAE del generador.

El inventario tras retirar la reanudación terminada fue 13,9553 GiB. El mejor modelo y los diez archivos de predicciones TransferTraj conservaron sus huellas. Ver inventario_conservacion.json; el tamaño puede crecer temporalmente con el punto de reanudación del entrenamiento activo.

<!-- historia_temporal_controlada -->
## Diagnóstico temporal y siguiente control

Con 9.198.270 decisiones locales reservadas de Train, añadir horario UTC de entrada a zona redujo NLL de 0,558328 a 0,541826. Diferencia −0,016501, IC95 % por día [−0,018592; −0,014138]. La edad de permanencia también aporta respecto a una probabilidad constante. Esto no es un resultado de MAE de viaje.

Los conteos y probabilidades por contexto ocupan 18.461.049 bytes, sin bancos por consulta. Se verificó que excluir una semana completa deja invariantes todos los niveles de respaldo. Una recuperación corrigió la comprobación de dimensiones sin recalcular ni cambiar los conteos; la nueva implementación reproduce el diagnóstico anterior con diferencia inferior a 0,000001 nats.

El siguiente paso condicionado compara cabezas equivalentes con y sin historial sobre la misma representación neuronal congelada. Se exige beneficio frente al control y frente al modelo neuronal original antes de iniciar otra GPU. Las semanas reservadas solo validan las nuevas cabezas: la representación ya vio Train. Ver plan_control_historia_condicional.json y pipeline_historia_estado.json. El entrenamiento temporal preparado no se ejecuta si falla ese control.

<!-- inferencia_canonica_nativo -->
## Estado de la evaluación nativa

Terminaron diez épocas y 79.960 actualizaciones; el checkpoint elegido fue la época 8, huella 04f84b39784f0fb039fb60c8ce83a9264ce71cbbdba1b6ad7260e846846a57fc. La evaluación original lenta se sustituyó sin repetir entrenamiento. La evaluación de reemplazo guarda avances de predicciones cada 8.192 filas y verifica orden y consultas individuales.

Los lotes 64 y 128 fueron más rápidos, pero cambiaron algunas trayectorias simuladas al cambiar el cálculo numérico. Se conservó el lote canónico 16: no se atribuye una aceleración. La variante temporal aún no iniciada fijó canónico 128 antes del entrenamiento y tiene su propio control inicial; esa es una configuración numérica distinta, sin afirmación de equivalencia con lote 16. La referencia principal conserva siempre su lote canónico 2.048.

La evaluación de los 127.941 viajes está en curso. El peso elegido en la muestra de selección es cero: equivale al modelo de referencia y no es una mejora aprendida. Se medirán también todas las otras salidas declaradas; todavía no se les atribuye un MAE completo.

Un instrumento adicional verificó por enumeración exacta el gradiente de CRPS condicional para muestras completas independientes aceptadas. Es una posible forma distinta de entrenar con el tiempo total generado, motivada por la meseta de MAE frente a la pérdida local descendente. No se ha implementado todavía su muestreador recurrente ni se ha entrenado en Porto. Ver plan_riesgo_total_generado.json.

<!-- riesgo_total_preparado -->
## Entrenamiento con el tiempo completo generado

Se implementó un muestreador que genera trayectorias completas y acepta las primeras muestras que cumplen el soporte fijo de 60 a 1.785 s. El tiempo real no forma parte de su interfaz. Conserva las probabilidades no condicionadas de cada trayectoria; el estimador de CRPS usa coeficientes de suma cero para cancelar el normalizador desconocido del condicionamiento. Requiere muestras independientes y al menos tres por consulta; no se aplica directamente a los bancos de probabilidades del evaluador.

La prueba sintética generó 32.768 trayectorias aceptadas para 4.096 consultas y coincidió con el gradiente analítico dentro de su error Monte Carlo. En cuatro consultas reales de Train se obtuvieron gradientes finitos y distintos de cero en las cabezas de ruta, permanencia y finalización y en la red recurrente. La prueba usó CPU, dejó los pesos intactos y no calcula un MAE predictivo.

El ensayo preparado optimiza el CRPS condicional de viajes generados y mantiene regularización por prefijos reales de Train. Usa ocho muestras aceptadas por consulta, 16 consultas por actualización acumulada, hasta 4.000 actualizaciones y una prueba GPU obligatoria antes del tramo largo. Un fallo de soporte utiliza la pérdida supervisada de esa misma consulta y se registra; la cohorte de evaluación permanece completa. Los conteos históricos excluyen todo el bloque Train correspondiente a la consulta.

El proceso se encadena después de los resultados completos y controles ya programados. Se registrarán sus conclusiones antes de iniciar otra mejora. Todavía no se ha ejecutado este entrenamiento ni existe un MAE suyo. Ver plan_riesgo_total_generado.json, sondeo_riesgo_total_cpu.json y pipeline_riesgo_estado.json. Test permanece cerrado y la meta <73,5 s sigue pendiente.
