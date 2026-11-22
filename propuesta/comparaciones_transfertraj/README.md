# Resultados locales relacionados con TransferTraj

Se conservan diez archivos originales de predicciones de validación, sin modificar sus bytes. Nueve corresponden a corridas terminadas; uno es un intento incompleto y no debe usarse como modelo entrenado. Entre las nueve corridas terminadas hay siete modelos de la familia TransferTraj y dos controles XGBoost con representaciones.

Todos usan la misma validación local de 127.941 viajes. No son resultados de Test ni una reproducción del benchmark publicado de Porto: la cohorte preprocesada de 323.481 viajes no estaba disponible.

| Variante | MAE (s) | Predicciones |
|---|---:|---|
| Modalidades normalizadas, sin preentrenamiento | 166.448130 | [archivo](od_modal_normalizado_sin_pre_s0/validacion.npz) |
| Reloj preciso, pérdida de duración MAE | 158.877006 | [archivo](od_reloj_preciso_duracion_mae_sin_normalizar_s0/validacion.npz) |
| Reloj preciso, pérdida de duración MSE | 175.778508 | [archivo](od_reloj_preciso_duracion_mse_sin_normalizar_s0/validacion.npz) |
| Reloj preciso, objetivo denominado oficial, sin preentrenamiento | 162.898605 | [archivo](od_reloj_preciso_oficial_sin_normalizar_s0/validacion.npz) |
| Intento inicial incompleto | 554.525875 | [archivo](od_transfertraj_adaptado_sin_pre_s0/validacion.npz) |
| Atención matemática, con preentrenamiento de 30 épocas | 171.720112 | [archivo](od_transfertraj_math_pre30_s0/validacion.npz) |
| Atención matemática, sin preentrenamiento | 164.367686 | [archivo](od_transfertraj_math_sin_pre_s0/validacion.npz) |
| Coordenadas métricas, sin preentrenamiento | 164.600961 | [archivo](od_transfertraj_metrico_sin_pre_s0/validacion.npz) |
| XGBoost con representación aleatoria: control adicional | 134.545698 | [archivo](od_xgboost_transfertraj_aleatorio_s0/validacion.npz) |
| XGBoost con representación preentrenada 6 épocas: control adicional | 134.300958 | [archivo](od_xgboost_transfertraj_prefijo6_s0/validacion.npz) |

Cada carpeta conserva las métricas disponibles, configuración, selección e historial. `indice.json` identifica estado, arquitectura, huellas y nombres de las matrices. `validacion.npz` contiene `row`, `pred`, `true` y, cuando estaban presentes, `timestamp` y cuantiles. Se comprobó su alineación y se recalculó el MAE antes de eliminar sus recursos pesados.

También se conservaron los metadatos de tres etapas de preentrenamiento, sin atribuirles una evaluación OD propia. Los checkpoints y recursos masivos de esas ramas se eliminaron; las fuentes históricas están en `../resultados/historial_evidencia.zip` y el código oficial consultado en `../referencias/TransferTraj`. Para comparaciones sobre esta validación no hace falta regenerarlos.
