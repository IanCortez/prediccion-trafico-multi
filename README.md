# Predicción de tráfico: avances para revisión de tesis

Rama de resultados y modelos, actualizada el 22 de noviembre de 2026.
El código original del alumno en src/ permanece intacto.

**Mejor MAE completo verificado: 129,990034 s en los mismos 127.941 viajes de Validation.**
La meta de confirmación <73,5 s sigue pendiente y Test permanece cerrado.

- [Qué cambió y cómo revisar el modelo de 129,99 s](revision_tesis/ACTUALIZACION_129_99.md).
- [Guía para Ian](revision_tesis/README.md).
- [Resultados y conclusiones de los experimentos](propuesta/RESUMEN_RESULTADOS.md).
- [Arquitectura vigente y límites científicos](propuesta/ARQUITECTURA.md).
- [Comparaciones locales con TransferTraj](propuesta/comparaciones_transfertraj/README.md).

La entrega contiene el nuevo checkpoint y sus dos padres, fuentes congeladas, métricas,
predicciones y evidencia de reproducción. Las memorias grandes de Train y mapas permanecen
como recursos externos identificados por SHA256. El ensamble histórico de 131,06 s conserva
su registro, pero tres de sus cuatro checkpoints se eliminaron antes de esta rama.
Cada actualización es una instantánea: los registros de procesos no describen ejecuciones
que sigan activas al consultar GitHub.
