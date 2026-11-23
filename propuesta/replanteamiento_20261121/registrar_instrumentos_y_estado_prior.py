from base_replanteo import *
import ast
p=OUT/'verificar_conservacion.py';s=p.read_text();marker='# Qualified geometryprior-only branches and their source-complete dependencies.'
if marker not in s:
 insert="""
# Qualified geometryprior-only branches and their source-complete dependencies.
file=OUT/'modelos_prior_arco_protegidos.json'
if file.exists():
 report['arclength_prior_candidates']=[]
 for candidate in read(file)['candidates']:
  path=Path(candidate['model_file']).resolve();assert path.is_relative_to(OUT.resolve()) and sha(path)==candidate['model_sha256']
  for name,digest in candidate['dependencies'].items():assert sha(OUT/name)==digest
  report['arclength_prior_candidates'].append(dict(id=candidate['id'],mae_s=candidate['validation']['mae_s'],unchanged=True))
"""
 s=s.replace("write(OUT/'inventario_conservacion.json'",insert+"\nwrite(OUT/'inventario_conservacion.json'");ast.parse(s);p.write_text(s)
p=OUT/'consolidar_auditoria_resultados.py';s=p.read_text();marker='# Geometryprior-only semanticcorrection, fixed six epochs.'
if marker not in s:
 insert="""
# Geometryprior-only semanticcorrection, fixed six epochs.
source=OUT/'prior_arco_subtrayectos_s0/resultado.json'
if source.exists():
 r=read(source);z=load_archive(source.parent/'validacion.npz')
 append(source.parent.name,'Subtrayectos: prior geométrico coherente',z['pred'],r['validation'],source)
"""
 s=s.replace('pending=[',insert+'\npending=[')
 s=s.replace("'mediana_directa_s0','atencion_historia_individual_s0'] if", "'mediana_directa_s0','atencion_historia_individual_s0','prior_arco_subtrayectos_s0'] if")
 ast.parse(s);p.write_text(s)
state=read(OUT/'CONTINUIDAD_OPERATIVA.json')
state['current_trial']=dict(experiment='prior_arco_subtrayectos_s0',stage='Sixfixedepochs, same34timefeatures/normalizer/NNparameters; corrected arclengthgeometryprioronly.',
 pid=35668,exec_session=18840,log='entrenar_prior_arco_subtrayectos.log',Test='sealed')
state['next_after_result']=['Inspectactualprocess/log/result; do not reruntraining on observationtimeouts.',
 'Auditconcluir_prior_arco.py then independentreplayifqualified. Comparewithtime6control andverifiedbest129.990034.',
 'IfTTmeaningfulgain, retainmatchedinitialization andsource semanticproofs; no causal/global73.5claim withoutindependentconfirmation.',
 'DoNOTrunsupersededentrenar_atencion_subtrayectos_arco.py (Arc34inputreplacement); currentcorrectorretainsalltimeinformation.',
 'Previousjoint24completed130.685980, qualifiedfullcheckpoint andindependentraw65/full127941replaypassed. Matchedsourcevaluecontrolnottriggered.']
state['resource_policy']='OneGPU .5cap; correctedprior TrainB512 measured5326MiB, canonical2048inference beforeTrainresident6090MiB. FullTrain tensors releasedbeforefinalevaluation.'
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
active=read(OUT/'arquitectura_activa.json');active.update(current_experiment='prior_arco_subtrayectos_s0',
 current_protocol='plan_prior_geometrico_arco.json',checkpoint_kind='Branchdelta ofprotected30componentparent; 34timefeaturesunchanged,auxiliaryarcgeometryprior',
 previous_joint_result=read(OUT/'subtrayectos_conjuntos_s0/resultado.json')['validation'])
write(OUT/'arquitectura_activa.json',active)
p=BASE/'ARQUITECTURA.md';s=p.read_text();s=s.replace('**Ensayo actual: entrenamiento conjunto desde cero.**','**Entrenamiento conjunto desde cero: cerrado.**')
needle='[TrajRoute, MDM 2025]'
if '## Corrección semántica del prior geométrico' not in s:
 index=s.index(needle)
 section="""## Corrección semántica del prior geométrico

El entrenamiento conjunto terminó sus 24 épocas en **130,685980 s** y no mejoró 129,990034 s. Su estado completo ocupa 18,39 MiB y se conserva por estar por debajo de 131,06 s. La reproducción recargó ese checkpoint y comprobó todos los cuantiles y las consultas crudas. El control condicional no se activó.

Una auditoría independiente identificó una inconsistencia: la supervisión geométrica original usa siete posiciones separadas por distancia de arco; la nueva fuente usaba posiciones separadas por tiempo como referencia geométrica. En 512 padres Train, el error mediano entre esas posiciones fue de 212,32 m. La longitud aproximada con nueve puntos también subestimaba un 10,75 % en la mediana. Esto no invalida el MAE del procedimiento congelado, pero impone a la red una corrección auxiliar innecesaria.

El muestreo corregido por arco coincide con la supervisión original con error mediano de 0,048 m y máximo de 0,973 m tras la representación float16. Se contrastaron además 892 intervalos parciales y se comprobó que añadir muestras estacionarias cambia el tiempo histórico, manteniendo la geometría. La distancia acumulada de Train ocupa 144,72 MiB; no se alteró la memoria original de puntos.

**Ensayo activo:** se conserva toda la representación temporal de 34 variables y su normalización original. Se aporta por separado la geometría por arco únicamente al prior del objetivo geométrico auxiliar. No se añaden parámetros neuronales. Un instrumento con pesos no nulos verificó que la distribución temporal es idéntica con pesos fijos y que sí cambia la referencia geométrica.

La comparación usa el mismo padre congelado de 30 componentes, idéntica inicialización neuronal, optimizador, orden de lotes y seis épocas fijas del control de 129,990034 s. Así se comprueba el efecto de corregir el aprendizaje auxiliar sin eliminar la información temporal histórica. No se alimenta la función corregida a los pesos antiguos.

La variante preparada que reemplazaba las 34 entradas por geometría de arco no llegó a ejecutarse y fue sustituida antes de evaluar un resultado. El ensayo vigente es prior_arco_subtrayectos_s0. Todas las consultas primarias, tiempos, filtros y campos públicos siguen iguales; Test permanece cerrado.

"""
 s=s[:index]+section+s[index:];p.write_text(s,encoding='utf-8')
p=BASE/'RESUMEN_RESULTADOS.md';s=p.read_text()
s=s.replace('El ensayo activo entrena desde cero y conjuntamente los componentes de rutas y subtrayectos: 24 épocas fijas, sin escoger una época de Validation. Sus controles se congelaron antes de observar la validación final.',
 'El ensayo conjunto de 24 épocas terminó en 130,685980 s y quedó conservado con reproducción completa. El ensayo activo corrige únicamente el prior geométrico auxiliar: conserva las 34 variables temporales, normalización, arquitectura e inicialización, durante seis épocas fijas.')
s=s.replace('y 41 checkpoints neuronales que cumplen el umbral de conservación.','y 42 checkpoints neuronales que cumplen el umbral de conservación.')
p.write_text(s,encoding='utf-8')
print('Currenttrial, preservation and semanticcorrection rationale updated.')
