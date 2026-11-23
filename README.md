# Predicción de tráfico: resultados y cierre de la investigación

**Cierre final: 23 de noviembre de 2026. Meta MAE <73,5 s no alcanzada. Test sin evaluar.**
La última propuesta terminó y la investigación se cerró por indicación del usuario.

**Mejor MAE completo verificado: 129,990034 s sobre 127.941 viajes de Validation.**
Se conservan el nuevo checkpoint y sus dos padres, el modelo original, las predicciones,
fuentes, protocolos y evidencia de reproducción.

- [Comenzar por el cierre final y sus conclusiones](revision_tesis/CIERRE_FINAL.md).
- [Informe final íntegro](propuesta/replanteamiento_20261121/RESUMEN_FINAL_ULTIMA_PROPUESTA.txt).
- [Guía de revisión para Ian](revision_tesis/README.md).
- [Modelo de 129,99 s: cadena de dependencias y limitaciones](revision_tesis/ACTUALIZACION_129_99.md).
- [Historial de resultados](propuesta/RESUMEN_RESULTADOS.md).
- [Arquitectura y límites científicos](propuesta/ARQUITECTURA.md).
- [Comparaciones locales con TransferTraj](propuesta/comparaciones_transfertraj/README.md).

La última red vial no superó su referencia en 10.000 consultas reservadas de Train:
131,9850 s frente a 129,6225 s. Son datos distintos de Validation primaria.
La reproducción pasó, pero no hubo una mejora predictiva útil.

El código original del alumno en src/ permanece intacto. Las memorias históricas grandes y
los mapas siguen siendo recursos externos identificados por SHA256. Los registros antiguos
de procesos son históricos; el documento CIERRE_ULTIMA_PROPUESTA.json fija el estado final.
