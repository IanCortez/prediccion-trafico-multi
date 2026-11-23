<!-- mejor_actual_subtrayectos_20261122 -->
## Mínimo numérico vigente: 129,990034 s

La atención sobre subtrayectos terminó seis épocas fijas. Su diferencia de −0,168007 s frente al padre tiene un intervalo exploratorio por día [−0,222310; −0,110332]. No alcanza el criterio práctico de 0,5 s ni confirma la meta de 73,5 s. Test sigue cerrado.

El checkpoint de 0,90 MiB guarda solo la rama nueva. Requiere atencion_historia_individual_s0, época 3, y su padre historica_fresca_s0, época 27. El cargador comprueba los pesos, fuentes y normalización. No se deben borrar estos antecesores ni los bancos necesarios. Las dependencias y sus huellas están en replanteamiento_20261121/nuevo_mejor_verificado.json y modelos_atencion_subtrayectos_protegidos.json.

Para recargar el checkpoint y reproducir exactamente los cuantiles de las 127.941 consultas y las 65 consultas crudas, invertidas e individuales:

~~~powershell
Set-Location -LiteralPath 'E:\UTEC\Tesis\TesisIan'
. '.\auditoria\entorno.ps1'
& $TaskPython -B '.\propuesta\replanteamiento_20261121\reproducir_atencion_subtrayectos.py'
~~~

La prueba escribe atencion_subtrayectos_s0/reproduccion_independiente.json. La clase FrozenSubtrajectoryExpert de inferencia_atencion_subtrayectos.py recibe listas de consultas con únicamente o_lon, o_lat, d_lon, d_lat y timestamp; devuelve cuantiles 10, 50 y 90 en segundos. Esta es una evaluación retrospectiva de Porto con recursos estáticos e históricos congelados.

El entrenamiento de esta rama se conserva en entrenar_atencion_subtrayectos.py y sus fuentes congeladas. Una recuperación separada completó la evaluación final con los mismos pesos, descargando Train de la memoria; no se repitió el entrenamiento. La variante conjunta desde cero es otro experimento y no reemplaza esta receta mientras no termine y se verifique.

Las instrucciones siguientes corresponden a la referencia original de 131,256712 s, que también permanece protegida.

# Reproducir la referencia original conservada

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
