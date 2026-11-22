# Protocolo congelado antes de entrenar la nueva propuesta

Versión 1, 2026-11-18. Objetivo principal: MAE < 73,5 s en Porto.
No se garantiza de antemano alcanzar un resultado de investigación.

## Datos y partición
CSV original conservado. SHA256 esperado: 2891d03ea99ce0fc203daedd434848a1a8baa7a8fb6da10f8d7ff26559efff6d.
Cohorte primaria: sin MISSING_DATA, 5–120 puntos, duración (n-1)*15 s, OD >=200 m, velocidad OD <=120 km/h y AMBOS extremos dentro de latitud [41.1405,41.1865], longitud [-8.6887,-8.5557]. Se corrige el límite superior 41.1859 del código por el 41.1865 de la tabla 8 final de TransferTraj. Estos filtros no reproducen necesariamente los 323.481 viajes del artículo.
No se filtran viajes por error del modelo. Se registra además si TODOS los puntos están dentro del área, solo para diagnóstico.
Orden estable por salida y fila original. 80% entrenamiento, 10% validación, 10% prueba. Viajes con idéntico segundo de salida en una frontera se asignan al segmento posterior para impedir solapamiento temporal.
Los identificadores de las muestras de la auditoría anterior se marcan. Se excluyen de la confirmación los viajes de prueba ya expuestos en esa auditoría, conservando sus cantidades en el manifiesto.
Features y etiquetas de prueba se guardan por separado. Los scripts de iteración NO cargan las etiquetas de prueba.

## Selección
Todas las iteraciones se seleccionan por MAE de validación en segundos. Entrenamiento usa solo el 80% inicial. No se reajusta con validación antes de la primera confirmación.
Los cambios quedan registrados con hipótesis, configuración, MAE/RMSE/MdAE, tiempo, semilla, tamaño, características y conclusión.
Primero referencias constante y potencia; luego boosting OD y residuo neuronal. La decisión de incorporar mapas o estado histórico debe justificarse por errores de VALIDACIÓN.
Se ejecutan semillas de confirmación para la configuración elegida. El acceso final a prueba requiere un archivo de candidato congelado con hashes de modelo, datos y configuración.
Si la prueba falla, no se la trata como validación para retocar y volver a certificar sobre los mismos viajes: una iteración posterior deberá declarar esa exposición y obtener un nuevo bloque confirmatorio o reconocer el resultado como exploratorio.

## Instrumentos
1. Manifiesto de datos, filtros, hashes y particiones.
2. Contrato de información: lista blanca OD/salida; prohibición de longitud/número de puntos, duración y ruta observada en características.
3. Pruebas de invariancia al modificar etiquetas reservadas, separación temporal y correspondencia de filas.
4. Registro de experimentos y bitácora crítica por corrida.
5. Curvas de aprendizaje / ablaciones de pérdida, base física, Fourier, rejilla y resolución.
6. Diagnóstico por distancia, hora y día; mejoras pareadas con bootstrap por día de salida, no solo por viaje.
7. Confirmación con intervalo de confianza, predicciones, métrica por semilla y ensemble declarado si se usa.
8. Comparabilidad: cohorte, datos etiquetados, mapas, entradas, hardware, costo y diferencia frente al artículo.
9. Para transferencia real: ganancias frente a target-only con la misma arquitectura y presupuesto TOTAL de etiquetas incluyendo validación, y frecuencia de transferencia negativa. No se sustituyen ciudades faltantes por datos sintéticos.

## Criterio de aceptación
MAE puntual de confirmación <73,5 s, unidades verificadas y predicciones finitas. Se informa además IC95% por bloques; una afirmación fuerte requiere que su extremo superior esté bajo 73,5.
Alcanzar la cifra en otra cohorte no demuestra superar TransferTraj en su benchmark exacto.

