# Arquitectura conservada

Entrada: cinco campos, `o_lon`, `o_lat`, `d_lon`, `d_lat`, `timestamp`. No se acepta trayectoria, duración ni número de puntos de la consulta.

Se calculan 216 características de geometría OD, reloj/calendario de Lisboa, contexto urbano, rutas calculadas sobre el mapa y perfiles históricos construidos con entrenamiento. Un TabM de 16 miembros y ancho 512 produce ocho componentes condicionales de duración sobre 116 valores entre 60 y 1.785 segundos. La predicción puntual es la mediana de la distribución combinada.

La forma de la trayectoria real se utiliza como supervisión auxiliar en entrenamiento, con peso 0,2. La consulta utiliza las rutas calculadas. El entrenamiento combina verosimilitud, puntuación de distribución acumulada y el objetivo geométrico. La selección original usó validación y la evaluación conservada fija float32 y lotes de 2.048 consultas.

La disponibilidad histórica de parte del contexto estático no está certificada; se trata de una evaluación retrospectiva por lotes. El resultado no establece superioridad sobre TransferTraj en su cohorte publicada.

El único checkpoint retenido está en `experimentos/od_mezcla8_geometria_conjunta_s0_fp32_v2`. Sus 115 archivos de fuentes congeladas se conservan completos. El historial de arquitecturas descartadas quedó en el archivo de evidencia, separado de esta implementación.
