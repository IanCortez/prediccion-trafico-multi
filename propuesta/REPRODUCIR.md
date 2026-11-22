# Reproducir el modelo conservado

Todo se ejecuta dentro de `E:\UTEC\Tesis\TesisIan`, con el entorno Conda local. Se mantienen los datos originales, los datos preparados necesarios, mapas y perfiles del modelo. Los dos archivos de distancias y los perfiles históricos se necesitan para construir entradas desde consultas OD nuevas.

Para comprobar el checkpoint, sus fuentes, las 127.941 predicciones de validación y las consultas crudas, sin entrenar ni evaluar Test:

```powershell
Set-Location -LiteralPath 'E:\UTEC\Tesis\TesisIan'
. '.\auditoria\entorno.ps1'
& $TaskPython -B '.\propuesta\reproducir_mejor.py'
```

El resultado se escribe en `propuesta/instrumentos/verificacion_modelo_conservado.json`. Se exige diferencia cero frente a las predicciones guardadas, incluidas 65 consultas OD, orden inverso y consulta individual. No se usa red.

La receta y las fuentes para volver a entrenar se conservaron. Solo si se decide ejecutar un entrenamiento nuevo, con un identificador que no exista:

```powershell
& $TaskPython -B '.\propuesta\reentrenar_mejor.py' --id 'reentrenamiento_manual_s0'
```

Ese comando inicia una única corrida de la receta original: semilla 0, máximo 100 épocas, lote 512, 16 miembros, ancho 512 y peso geométrico 0,2. El lanzador adapta únicamente copia de fuentes y registro a la estructura depurada. No reactiva las antiguas colas. La comprobación realizada durante la limpieza es de inferencia; no se ejecutó un reentrenamiento ni se promete identidad bit a bit de un entrenamiento nuevo.

Los datos de prueba y su máscara de exposición permanecen guardados. Estos comandos no evalúan Test. El entorno se documenta en `auditoria/conda-explicito.txt` y `auditoria/dependencias-exactas.txt`.
