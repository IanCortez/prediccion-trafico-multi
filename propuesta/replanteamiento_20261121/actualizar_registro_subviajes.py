from base_replanteo import *
path=OUT/'consolidar_auditoria_resultados.py';code=path.read_text(encoding='utf-8-sig')
old="""  append(run,label+' — mejor época',z['pred'],r['validation'],source,kind='fresh_selected')
pending=[run for run in ['campo_multiescala_s0'] if not (OUT/run/'resultado.json').exists()]"""
new="""  append(run,label+' — mejor época',z['pred'],r['validation'],source,kind='fresh_selected')
  for row in r['all_epoch_validation']:
   epoch=row['epoch'];append(run,label+' — época '+str(epoch),z['epoch_'+str(epoch)],row['validation'],source,kind='epoch_checkpoint')
for run,label in [('aux_subviajes_s0','Supervisión auxiliar de subtrayectos'),('control_aux_subviajes_s0','Control auxiliar de subtrayectos')]:
 source=OUT/run/'resultado.json'
 if source.exists():
  r=read(source);z=load_archive(source.parent/'validacion.npz')
  append(run,label+' — inicial',z['epoch_0'],reference_metric,source,kind='initial_reference')
  for row in r['all_epoch_validation']:
   epoch=row['epoch'];append(run,label+('' if epoch==6 else ' — época '+str(epoch)),z['epoch_'+str(epoch)],row['validation'],source,kind='fixed_final_epoch' if epoch==6 else 'epoch_checkpoint')
pending=[run for run in ['campo_multiescala_s0','aux_subviajes_s0'] if not (OUT/run/'resultado.json').exists()]"""
assert old in code;path.write_text(code.replace(old,new),encoding='utf-8')
text="""

<!-- cierre_campo_y_subviajes_supervisados_20261122 -->
## Cierre del campo espacial y siguiente supervisión

El campo multiescala terminó20épocas: su mejor resultado fue135,423828s en época4. Después la pérdida de Train siguió bajando mientras la validación empeoró, hasta150,196966s en época20. Esta configuración sobreajusta y no supera la referencia131,256712s. El control adicional de capacidad se omitió según la regla fijada. Se conservan sus métricas y todas las predicciones por época; sus pesos temporales descartados se retiraron.

La siguiente prueba añade una tarea supervisada con tiempos de subtrayectos exclusivamente de Train:250.000 viajes elegidos con semilla fija produjeron475.076 ejemplos. Para observaciones i,j, la etiqueta es15*(j-i)s y la salida es la fecha del viaje más15*i. Las entradas siguen siendo únicamente origen, destino y fecha de salida. La procedencia, los índices de observación y la trayectoria no entran al modelo. No se cambian filtros ni etiquetas de la cohorte primaria.

Se usa el codificador del mejor modelo con dos salidas de duración: la original para viajes completos y otra para subtrayectos, con ocho componentes y parámetros independientes por miembro. La salida auxiliar evita exigir que los subtrayectos tengan el mismo tiempo de espera que los viajes completos. Primero se ajusta esa salida con el codificador congelado durante una época; después se entrenan seis épocas conjuntas, peso auxiliar0,2, tasa1e-5 para el modelo original y0,001 para la salida auxiliar. Solo si aparece una mejora se ejecutará el control equivalente con peso auxiliar cero.

Cada viaje padre llega antes del primer viaje de validación. Los padres que cruzan una semana UTC se excluyen solo de la tarea auxiliar: así se puede excluir su semana completa de todos los niveles de históricos de su propia consulta. Los cinco datos de entrada reconstruyeron exactamente las216características para65viajes de Train antes de crear el conjunto auxiliar. Las10000pruebas de intervalos y el rechazo de campos de procedencia pasaron. El entrenamiento debe superar además la prueba de gradientes y reproducir la referencia antes y después del calentamiento.

La preparación conserva consultas, etiquetas y procedencia en11.116.354bytes. El caché de características tiene un límite de600MiB y se retirará si el ensayo terminado no mejora; se conserva el procedimiento para reconstruirlo. Se guardan métricas y predicciones de **todos** los experimentos, incluso con MAE mayor de131,06s. Los pesos del mejor modelo y los candidatos por debajo de131,06s permanecen protegidos. Los resultados TransferTraj y los datos originales también.

Ver plan_aux_subviajes.json, pipeline_subviajes_estado.json y conclusion_aux_subviajes.json cuando exista. Estado todavía exploratorio: objetivo73,5s pendiente y Test cerrado.
"""
# Use the exact observed epoch20 metric, do not copy an approximate remembered number.
field=read(OUT/'campo_multiescala_s0/resultado.json')
last=field['all_epoch_validation'][-1]['validation']['mae_s']
text=text.replace('150,196966',f'{last:.6f}'.replace('.',','))
for dest in [OUT/'README.md',BASE/'RESUMEN_RESULTADOS.md']:
 oldtext=dest.read_text(encoding='utf-8-sig')
 assert '<!-- cierre_campo_y_subviajes_supervisados_20261122 -->' not in oldtext
 dest.write_text(oldtext+text,encoding='utf-8')
print('Updated full-output audit and current experiment notes.')
