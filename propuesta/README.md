# Tesis de Ian: proyecto depurado

Mejor modelo conservado: **MAE 131,256712 s en validación de 127.941 viajes**. La meta de Test <73,5 s sigue sin demostrarse. Las ejecuciones automáticas están detenidas.

- [Resumen de resultados](RESUMEN_RESULTADOS.md).
- [TransferTraj: métricas, predicciones y límites de comparación](comparaciones_transfertraj/README.md).
- [Reproducción del modelo](REPRODUCIR.md).
- [Arquitectura conservada](ARQUITECTURA.md).
- [Protocolo original](PROTOCOLO.md).

Se conserva un solo checkpoint: `experimentos/od_mezcla8_geometria_conjunta_s0_fp32_v2`. Las fuentes congeladas y la procedencia de su selección están dentro de esa carpeta. `datos` contiene los recursos que necesita; `referencias` conserva mapas y el código oficial de TransferTraj consultado. `resultados` contiene tablas y un archivo pequeño de evidencia histórica desduplicada.

Los datos originales siguen en `../tesis`, el clon original en `../codigo-ian`, el entorno local en `../.conda-env` y los inventarios de limpieza en `../auditoria/limpieza`. No se borraron los originales ni se abrieron las etiquetas de Test para evaluar un modelo.
