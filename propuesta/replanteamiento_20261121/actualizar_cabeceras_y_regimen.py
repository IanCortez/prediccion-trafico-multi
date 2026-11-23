from base_replanteo import *
best=read(OUT/'nuevo_mejor_verificado.json');assert best['id']=='atencion_subtrayectos_s0'
p=BASE/'RESUMEN_RESULTADOS.md';s=p.read_text();a=s.index('<!-- estado_actual_verificado_inicio -->');b=s.index('<!-- estado_actual_verificado_fin -->')+len('<!-- estado_actual_verificado_fin -->')
head="""<!-- estado_actual_verificado_inicio -->
Mínimo numérico vigente: **MAE 129,990034 s**, en los mismos 127.941 viajes de validación. Corresponde a la atención sobre subtrayectos, época final fija 6. Se recargaron sus pesos y reprodujeron todos los cuantiles, más 65 consultas crudas con inversión de orden y caso individual.

Frente al padre de 130,158042 s, la diferencia es −0,168007 s, con intervalo exploratorio por día [−0,222310; −0,110332]. Es una mejora pequeña; RMSE empeora ligeramente y no se cumple el criterio predeclarado de mejora práctica de 0,5 s. La búsqueda adaptativa y el uso repetido de Validation impiden considerarla confirmación.

**La meta de confirmación <73,5 s sigue pendiente. Test está cerrado.** La cohorte publicada de TransferTraj es distinta de la local. Se conservan todos los resultados, incluidos los superiores a 131,06 s, las comparaciones TransferTraj y 41 checkpoints neuronales que cumplen el umbral de conservación. También se protege la fórmula no paramétrica de 130,959974 s con sus fuentes.

El ensayo activo entrena desde cero y conjuntamente los componentes de rutas y subtrayectos: 24 épocas fijas, sin escoger una época de Validation. Sus controles se congelaron antes de observar la validación final. El estado del proceso está en replanteamiento_20261121/CONTINUIDAD_OPERATIVA.json; el mejor y sus dependencias, en nuevo_mejor_verificado.json; la receta vigente, en REPRODUCIR.md. La última revisión de conservación registró 16,71 GiB.

Los resultados históricos siguientes conservan su contexto; las cifras de cohortes internas o con trayectoria real no cuentan como MAE OD primario.
<!-- estado_actual_verificado_fin -->"""
s=s[:a]+head+s[b:]
s=s.replace('El mejor modelo tiene 4.280.768 parámetros y 216 características','El modelo de referencia de 131,256712 s tiene 4.280.768 parámetros y 216 características')
title='## Instrumento Train: regímenes explícitos de duración'
if title not in s:
 s+='\n\n'+title+'\n\nEn la cohorte cronológica interna reutilizada de 40.000 consultas de ajuste y 10.000 reservadas, un bosque de distribuciones con puerta para T ≥ 900 s obtuvo 138,604500 s; su control global, 139,332000 s, y la referencia previa sobre esas mismas filas, 133,355393 s. La mejora de 0,727500 s frente al bosque global fue insuficiente y quedó 5,249107 s por detrás de la referencia. La puerta alcanzó AUC 0,830398. Se descarta esta implementación; no se barre el umbral ni se ejecuta una prueba primaria solo por el AUC. Métricas y predicciones conservadas. No se usaron Validation ni Test en este instrumento.\n'
p.write_text(s,encoding='utf-8')
p=BASE/'REPRODUCIR.md';s=p.read_text();marker='# Reproducir el modelo conservado'
b=s.index(marker)
head="""<!-- mejor_actual_subtrayectos_20261122 -->
## Mínimo numérico vigente: 129,990034 s

La atención sobre subtrayectos terminó seis épocas fijas. Su diferencia de −0,168007 s frente al padre tiene un intervalo exploratorio por día [−0,222310; −0,110332]. No alcanza el criterio práctico de 0,5 s ni confirma la meta de 73,5 s. Test sigue cerrado.

El checkpoint de 0,90 MiB guarda solo la rama nueva. Requiere atencion_historia_individual_s0, época 3, y su padre historica_fresca_s0, época 27. El cargador comprueba los pesos, fuentes y normalización. No se deben borrar estos antecesores ni los bancos necesarios. Las dependencias y sus huellas están en replanteamiento_20261121/nuevo_mejor_verificado.json y modelos_atencion_subtrayectos_protegidos.json.

Para recargar el checkpoint y reproducir exactamente los cuantiles de las 127.941 consultas y las 65 consultas crudas, invertidas e individuales:

~~~powershell
Set-Location -LiteralPath 'E:\\UTEC\\Tesis\\TesisIan'
. '.\\auditoria\\entorno.ps1'
& $TaskPython -B '.\\propuesta\\replanteamiento_20261121\\reproducir_atencion_subtrayectos.py'
~~~

La prueba escribe atencion_subtrayectos_s0/reproduccion_independiente.json. La clase FrozenSubtrajectoryExpert de inferencia_atencion_subtrayectos.py recibe listas de consultas con únicamente o_lon, o_lat, d_lon, d_lat y timestamp; devuelve cuantiles 10, 50 y 90 en segundos. Esta es una evaluación retrospectiva de Porto con recursos estáticos e históricos congelados.

El entrenamiento de esta rama se conserva en entrenar_atencion_subtrayectos.py y sus fuentes congeladas. Una recuperación separada completó la evaluación final con los mismos pesos, descargando Train de la memoria; no se repitió el entrenamiento. La variante conjunta desde cero es otro experimento y no reemplaza esta receta mientras no termine y se verifique.

Las instrucciones siguientes corresponden a la referencia original de 131,256712 s, que también permanece protegida.

"""
s=head+s[b:];s=s.replace('# Reproducir el modelo conservado','# Reproducir la referencia original conservada',1)
p.write_text(s,encoding='utf-8')
r=read(OUT/'piloto_regimen_duracion_Train_s0/resultado.json')
write(OUT/'conclusion_regimen_duracion_Train.json',dict(complete=True,pilot_metrics=r['metrics'],global_control=r['global_density_control'],
 reference=r['reference'],comparisons=r['comparisons'],gate_passed=False,
 conclusion='Explicit900sregime forest modestlyimproved itsglobalforestcontrol butwasworse thanexistingODreference. Retiretestedfactorization; no threshold/treecount sweep orfullprimaryrun.',
 limitations='ReusedheldTrain, limitedforestfamily, no universalclaimthat durationregimescannothelp.',
 all_metrics_predictions_sources_retained=True,weights_discarded=True,Test='sealed',goal='ACTIVE'))
print('Currentbest pointers, reproductionrecipe and negativeTrainpilot conclusion updated.')
