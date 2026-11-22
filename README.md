# Predicción de tráfico: avances para revisión de tesis

Esta rama conserva el código original del alumno y añade una instantánea de investigación para su revisión.
La guía de entrada es [revision_tesis/README.md](revision_tesis/README.md).

- [Resultados, modelos y límites de interpretación](revision_tesis/README.md).
- [Resumen de experimentos](propuesta/RESUMEN_RESULTADOS.md).
- [Comparaciones locales con TransferTraj](propuesta/comparaciones_transfertraj/README.md).
- [Arquitectura del modelo conservado](propuesta/ARQUITECTURA.md).
- [Nuevo replanteamiento y experimentos negativos](propuesta/replanteamiento_20261121/README.md).
- [Evidencia del ensamble histórico de 131,06 s](revision_tesis/ensamble_historico_131_06.json).

**Meta MAE <73,5 s: pendiente.** Mejor modelo individual conservado: **131,256712 s** en
127.941 viajes de Validation. El ensamble histórico llegó a **131,063381 s**, pero tres
de sus cuatro checkpoints se eliminaron en la limpieza previa. Se conserva su registro; no se afirma
que ese ensamble pueda cargarse desde esta rama. Test no fue evaluado.

Esta entrega es una instantánea: los estados de procesos y las cifras parciales son históricos.
