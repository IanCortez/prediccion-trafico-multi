from base_replanteo import *
p=BASE/'RESUMEN_RESULTADOS.md'
s=p.read_text(encoding='utf-8-sig')
s=s.replace('La meta MAE de prueba <73,5 s no se alcanzó. La prueba final permanece cerrada. No hay experimentos activos.','La meta MAE de confirmación <73,5 s todavía no se ha alcanzado. Test permanece cerrado y el trabajo continúa; el estado vigente está en el apartado de replanteamiento continuo.')
s=s.replace('Las iteraciones posteriores no justificaron continuar aumentando el costo.','En el ciclo histórico previo a la reanudación, las iteraciones posteriores no justificaron aumentar el costo de aquellos mecanismos.')
row='| Dirección y prior de conteos | 137,538866 s |'
if '| Permanencias por fases y referencia fuerte |' not in s:s=s.replace(row,row+'\n| Permanencias por fases y referencia fuerte | 132,700776 s |')
s=s.replace('Está entrenando una corrección física de la distribución del modelo conservado, con permanencias por fases, lotes de8 y acumulación2. Reproduce exactamente las127.941 predicciones originales antes del ajuste; sus distribuciones de referencia se mantienen solo en RAM. Este último se justifica con semanas reservadas de Train: el acierto de siguiente zona sube de 72,75 % a 76,37 %. Es evidencia sobre transiciones, no una mejora demostrada de duración.','La corrección física con permanencias por fases terminó 6.000 actualizaciones, con 96.000 consultas para el tiempo total y 384.000 estados locales. Obtuvo 132,700776 s, peor que la referencia en 1,444064 s (IC95 % exploratorio por día: 1,236382 a 1,665372 s). La rama espacial obtuvo 141,159167 s: mejoró dentro de su familia, pero sigue por encima de la referencia. Se reprodujeron exactamente las 127.941 predicciones de referencia antes del ajuste; las distribuciones se mantienen solo en RAM. Se realiza una única calibración declarada de antemano; no se repetirán búsquedas de pesos para esa misma rama congelada.')
marker='<!-- transicion_prefijo_nativo -->'
if marker not in s:
    s+='\n'+marker+'\nEl piloto de contexto compartido por viaje no justificó un entrenamiento completo: la diferencia de NLL frente a un contexto independiente por tramo fue +0,000183 por observación (IC95 % por día: −0,000264 a +0,000635), en semanas de Train reservadas solo para esos nuevos parámetros escalares. No es una prueba de independencia general. La siguiente alternativa es un generador recurrente de zona, permanencia y finalización a intervalos nativos de 15 s. En consulta usará únicamente su propia trayectoria generada y un horizonte fijo de 120 pasos, sin conocer la longitud real. Pasó pruebas de causalidad, gradientes, conservación de probabilidades y reconstrucción exacta de 1.000 viajes de Train. Todavía no tiene un MAE. Se espera la salida real del proceso GPU anterior para iniciar el entrenamiento.\n\nEl estado vigente de las ejecuciones está en pipeline_prefijo_estado.json y CONTINUIDAD_OPERATIVA.json, dentro de replanteamiento_20261121. Las métricas parciales o de selección no sustituyen los 127.941 viajes de Validación.\n'
p.write_text(s,encoding='utf-8')
p=OUT/'README.md'
s=p.read_text(encoding='utf-8-sig')
heading='## Ejecución activa'
assert heading in s
s=s[:s.index(heading)]+'''## Ejecución vigente

El entrenamiento con permanencias por fases terminó en 132,700776 s de MAE en los 127.941 viajes; la rama espacial obtuvo 141,159167 s. La diferencia pareada del modelo integrado frente a la referencia fue +1,444064 s, IC95 % exploratorio por día [1,236382; 1,665372]. No mejora al modelo conservado.

La calibración única declarada en protocolo evalúa ocho pesos sin repetir la generación. Elegir el peso cero equivale al modelo anterior y no constituye un nuevo resultado aprendido. Las predicciones distintas y todas sus métricas se conservan.

El piloto CPU no respalda compartir un único efecto de demora por todo el viaje frente a un efecto independiente por tramo: diferencia de NLL +0,000183, IC95 % [−0,000264; +0,000635]. Por ello no se inicia un entrenamiento completo de esa hipótesis. Las pruebas sintéticas de ese instrumento sí pasaron; un instrumento correcto puede dar un resultado negativo en los datos.

La siguiente propuesta es un generador recurrente de celdas a la frecuencia nativa de 15 s. Aprende a desplazarse, permanecer y terminar con prefijos reales de Train. Al consultar, la historia y el tiempo transcurrido son los que el propio generador simula; nunca se le entrega la longitud o la trayectoria real. Se acumula probabilísticamente la finalización de 128 partículas en cada paso de un horizonte fijo de 120 pasos. La semilla depende solo de OD/salida, partícula y paso.

Las pruebas CPU pasaron: equivalencia de procesamiento secuencial y por lote, causalidad ante cambios del futuro, independencia del relleno, gradientes finitos, absorción conocida y conservación de masa. En 1.000 viajes reales de Train, duración y extremos coinciden exactamente. El estado recurrente y la estimación por partículas añaden riesgo de desajuste entre entrenamiento y generación y variación Monte Carlo; no se presupone una mejora del MAE. UNITE (KDD 2026) es un antecedente de generación recurrente; este desarrollo no se presenta como réplica exacta ni como novedad por sí solo.

El supervisor continuar_a_prefijo.py espera mediante identificadores originales del sistema operativo a que terminen la calibración y su supervisor. Así evita confundir un PID posteriormente reutilizado con el proceso que terminó. Solo inicia un trabajo GPU.

Consultar pipeline_prefijo_estado.json, CONTINUIDAD_OPERATIVA.json, transicion_a_prefijos.json y los registros de cada ejecución. Los identificadores escritos en apartados históricos no indican procesos vivos.

El objetivo de confirmación <73,5 s permanece activo. Test continúa cerrado. Se conserva cada candidato con MAE completo <131,06 s, junto con el mejor modelo en todo caso, los datos originales y las métricas y predicciones TransferTraj. Un ajuste que termina o falla no cierra la investigación.
'''
p.write_text(s,encoding='utf-8')
write(OUT/'arquitectura_activa.json',read(OUT/'transicion_a_prefijos.json')['next'])
print('Resumen y arquitectura actualizados; Test cerrado.')
