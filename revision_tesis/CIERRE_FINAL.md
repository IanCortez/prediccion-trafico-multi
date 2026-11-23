# Cierre final para revisión de la tesis

Cierre del 23 de noviembre de 2026, solicitado después de terminar la última propuesta.
**La investigación experimental quedó cerrada. No se alcanzó MAE <73,5 s. Test permanece
sin evaluar.** El registro final indica que no quedan experimentos activos.

## Mejor resultado principal

**129,990034 s en los mismos 127.941 viajes de Validation**, del modelo de atención sobre
subtrayectos históricos. Se mantienen su checkpoint, los dos padres requeridos, predicciones,
fuentes y evidencia de reproducción completa. La mejora frente a su padre inmediato fue
0,168007 s: pequeña, con una sola semilla y selección adaptativa; no cumplió el criterio
práctico de 0,5 s. No es un resultado de confirmación independiente.

El ensamble histórico de 131,063381 s sigue documentado con sus componentes incompletos.
El modelo original de 131,256712 s y los archivos de predicciones TransferTraj se conservan.

## Última comparación: resultado negativo

Las tres cifras siguientes corresponden a **10.000 consultas cronológicas reservadas dentro
de Train**. No deben compararse directamente con los 129,99 s de Validation primaria.

| Variante | MAE (s) | RMSE (s) |
|---|---:|---:|
| Referencia pública congelada | **129,6225** | 204,4502 |
| Red vial con posición y compuerta | 132,6075 | 207,6219 |
| Red vial con movimiento y compuerta | 131,9850 | 206,7393 |

La red con movimiento empeoró 2,3625 s frente a la referencia, IC95% exploratorio por día
[1,4465; 3,3752]. Frente a la red de posición mejoró 0,6225 s, pero el intervalo
[-1,5102; 0,2410] incluye cero. Ninguna variante pasó los criterios predeclarados:
5 s frente a la referencia y, para movimiento, 2 s adicionales frente al control de posición.
No se amplió esta familia a Validation primaria ni se hicieron nuevas búsquedas después del resultado.

La red aislada, antes de la compuerta, obtuvo 280,248 s para posición y 268,386 s para movimiento.
La cobertura de red del 99,24 % no se tradujo en precisión. Las compuertas dieron un peso medio
cercano al 2 % a la red, pero aun así empeoraron la referencia. Una explicación plausible es que
condicionar los extremos no sustituye una política de rutas específica del destino; no se aisló
esa causa experimentalmente. También se documenta la asimetría de ajuste entre referencia y red.

## Qué puede sostenerse

- La mejor propuesta de atención aporta una mejora numérica pequeña en validación reutilizada.
- Modelar calles, movimiento, geometría o esperas no garantizó mejorar el tiempo total OD.
- Las pruebas de integridad y reproducción pasaron; su éxito no demuestra utilidad predictiva.
- Los pilotos de Train, los diagnósticos con información privilegiada y las cotas de familias
  fijas no acreditan el objetivo ni un límite del error irreducible.
- Los 73,5 s de TransferTraj proceden de otra cohorte procesada de Porto no disponible aquí.
- Faltan confirmación independiente y repetición entre semillas para establecer generalización.

## Evidencia y revisión

- [Informe final íntegro](../propuesta/replanteamiento_20261121/RESUMEN_FINAL_ULTIMA_PROPUESTA.txt).
- [Registro formal de cierre](../propuesta/replanteamiento_20261121/CIERRE_ULTIMA_PROPUESTA.json).
- [Resultado de la última comparación](../propuesta/replanteamiento_20261121/red_cinetica_Train_s0/resultado.json).
- [Conclusiones completas y limitaciones](../propuesta/replanteamiento_20261121/red_cinetica_Train_s0/conclusion.json).
- [Reproducción independiente registrada](../propuesta/replanteamiento_20261121/red_cinetica_Train_s0/reproduccion_independiente.json).
- [Mejor modelo y sus dependencias](ACTUALIZACION_129_99.md).

Desde la raíz de la rama:

    python -B revision_tesis/verificar_resultados.py
    python -B revision_tesis/verificar_nuevo_modelo.py
    python -B revision_tesis/verificar_cierre.py

El último comando recalcula los MAE de la comparación final, valida alineación, distribuciones
guardadas y huellas de los pesos. No reentrena, no ejecuta GPU, no abre Test ni sustituye la
reproducción de inferencia independiente ya registrada. Los datos grandes para inferencia
completa siguen siendo recursos externos, igual que en las entregas previas.

Se incluyen los dos pesos finales de compuerta de esta comparación, su constructor y el peso
de la referencia pública congelada. Sus resultados son de Train reservado y no sustituyen al
mejor modelo primario. Las copias originales de la investigación permanecen intactas.
