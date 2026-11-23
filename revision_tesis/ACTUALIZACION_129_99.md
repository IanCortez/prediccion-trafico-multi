# Actualización: mínimo verificado de 129,990034 s

Evaluación completa de los mismos 127.941 viajes de Validation. El modelo
atencion_subtrayectos_s0 terminó seis épocas fijas y su checkpoint final se recargó
para reproducir todos los cuantiles, 65 consultas desde cinco campos, orden inverso
y consulta individual. Test permanece cerrado.

| Referencia | MAE (s) |
|---|---:|
| Ensamble histórico, pesos incompletos | 131,063381 |
| Modelo original conservado | 131,256712 |
| Mezcla con proyección incierta | 131,125167 |
| Mezcla histórica entrenada desde cero | 130,204821 |
| Atención sobre viajes históricos completos | 130,158042 |
| **Atención sobre subtrayectos históricos** | **129,990034** |
| Entrenamiento conjunto desde cero | 130,685980 |
| Corrección del prior por distancia de arco | 129,994490 |

La mejora frente al padre inmediato es 0,168007 s, con intervalo exploratorio por día
[-0,222310; -0,110332] s. No pasa el criterio práctico predeclarado de 0,5 s. RMSE empeora
ligeramente. La selección adaptativa de la investigación no queda corregida por ese
intervalo. La meta de confirmación <73,5 s sigue pendiente.

## Pesos y dependencias

El nuevo archivo de 0,90 MiB contiene solo una rama adicional. **No es un modelo
independiente**. Se publican también sus dos padres:

    propuesta/replanteamiento_20261121/historica_fresca_s0/epoca_27.pt
    propuesta/replanteamiento_20261121/atencion_historia_individual_s0/epoca_3.pt
    propuesta/replanteamiento_20261121/atencion_subtrayectos_s0/epoca_6.pt

La cadena produce 38 componentes: 22 de rutas/prototipos, ocho de viajes completos
y ocho de subtrayectos. Las fuentes congeladas, normalizadores, protocolos, predicciones,
registros de protección y reproducción están incluidos. Otros checkpoints finales
conservados se enumeran en modelos_actualizados.json. Las épocas intermedias adicionales
permanecen protegidas en el proyecto local; no se duplican todas en este commit.

Para revisar la evidencia sin GPU ni memorias grandes:

    python -B revision_tesis/verificar_resultados.py
    python -B revision_tesis/verificar_nuevo_modelo.py

El segundo instrumento comprueba pesos, sus huellas, las seis/ tres/27 épocas requeridas
y el MAE completo guardado. No genera nuevas predicciones. La certificación de inferencia
completa proviene del registro de reproducción independiente que se incluye.

La inferencia desde consultas nuevas necesita, además, mapas/perfiles originales y
memorias de Train que no se suben a Git. recursos_externos_129_99.json identifica los
archivos y SHA256. El banco de puntos ocupa 253.502.723 bytes y excede el límite por archivo
de GitHub. No se presenta esta rama como una distribución autónoma de esos datos.
Los scripts históricos conservan rutas de la máquina original: deben adaptarse en una copia
sin alterar las fuentes congeladas o sus huellas. No se cambiaron las ejecuciones activas.

Los nuevos pilotos de 10.000/512 consultas Train y los diagnósticos privilegiados se
documentan por separado; no son nuevas mediciones de Validation primaria.
