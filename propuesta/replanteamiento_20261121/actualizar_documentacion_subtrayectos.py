from base_replanteo import *
p=BASE/'ARQUITECTURA.md';s=p.read_text()
s=s.replace('## Alternativa en evaluación: progreso vial recurrente','## Progreso vial recurrente: ensayo cerrado sin mejora')
needle='## Instrumentos y criterio de aceptación'
insert="""## Resultados recientes y replanteamiento

La consistencia del tiempo agregado se probó por separado: ajustó tiempos locales con todos los segmentos útiles de Train, manteniendo congeladas la política y la representación recurrente. El error del diagnóstico privado con recorrido conocido bajó de 95,36 a 76,29 s en 512 consultas Train. En las 127.941 consultas primarias, con recorrido generado y una etapa final idéntica, obtuvo **135,996132 s**, frente a 136,111205 s sin esa corrección y 135,988501 s del control de 216 variables. La mejora privada no se trasladó a la tarea OD.

Se probó también una alternativa global sin generación secuencial: difusión conjunta de la duración y una firma geométrica de 90 dimensiones. Recibió las mismas 216 características públicas y conservó todos los tiempos originales. Tras 30 épocas fijas, 32 pasos inversos y 64 muestras por consulta, obtuvo **139,953394 s**. No se eligieron otras épocas, cantidades de muestras o dimensiones después del resultado. Es una adaptación propia; no se presenta como reproducción de DOT.

La nueva memoria conserva los 42.624.621 puntos GPS de todos los padres Train autorizados en 241,76 MiB. Para una consulta busca puntos próximos al origen y destino, intersecta sus viajes y exige que el origen histórico preceda al destino al menos cuatro muestras. Utiliza el tiempo observado **15 × (j − i)** de cada fuente, ajustado únicamente por la relación de distancias OD. No conoce el recorrido, número de muestras o duración de la consulta.

El kernel directo con respaldo obtuvo **163,945881 s**; la mezcla con confianza según cantidad efectiva de fuentes y cercanía obtuvo **130,959974 s**. Ambos fallaron frente a 130,158042 s. La segunda fórmula queda conservada íntegramente porque cumple el umbral de conservación de 131,06 s, aunque no reemplaza al mejor modelo. Un 34,30 % de las consultas de Validation no encontró fuentes con este procedimiento. Se conservan también sus predicciones y métricas completas.

**Ensayo siguiente:** una rama de atención que aprende con Train el ajuste entre tiempos de subtrayectos y viajes completos. Recibe 64 fuentes, cada una con 34 variables: geometría de siete puntos intermedios, tiempo histórico, diferencias entre extremos, calendario y contexto de comienzo/final del viaje fuente. Ese contexto permite comprobar la hipótesis de que un intervalo interno no representa directamente la distribución de un viaje completo.

El mejor modelo queda congelado como padre de 30 componentes. Se añaden ocho componentes de atención, con normalización y anclaje temporal obtenidos exclusivamente de 32.768 consultas Train elegidas antes del entrenamiento. Se fijan seis épocas; se evalúa el estado final, sin escoger una época de Validation. Las fuentes de cada consulta Train excluyen su grupo semanal **antes** de construir el índice espacial, para que tampoco la cobertura de los vecinos dependa de la geometría excluida. Los tokens se reconstruyen al consultar: se guardan únicamente referencias compactas a padres y posiciones, sin bancos de distribuciones ni pares de puntos cuadráticos.

[TrajRoute, MDM 2025](https://www.johnkrumm.net/_files/ugd/3e3382_9d657170e9144dcb873791573d75c542.pdf) aporta un antecedente para seguir y combinar trayectorias históricas. Su evaluación de rutas contra Azure Maps y su ventana temporal próxima a la consulta no equivalen a nuestro pronóstico OD retrospectivo con Train cerrado. La propuesta actual se limita a intervalos históricos completos; no se afirma haber reproducido su buscador de rutas ni sus resultados.

"""
assert needle in s
if '## Resultados recientes y replanteamiento' not in s:s=s.replace(needle,insert+needle)
s=s.replace('y el entrenamiento de tiempos locales sin imponer todavía que su suma reproduzca todo el viaje.',
 'y la diferencia entre un intervalo interno y un viaje completo. La consistencia agregada ya se probó y no resolvió el error de la tarea OD.')
p.write_text(s,encoding='utf-8')
r=read(OUT/'memoria_subtrayectos_cdf_s0/resultado.json')
write(OUT/'conclusion_memoria_subtrayectos.json',dict(complete=True,main=r['validation'],control=r['metrics']['kernel_backoff'],
 paired_vs_protected=r['paired_vs_reference'],meaningful_gate_passed=False,
 conclusions=['Directhistoricalsegmenttimes do nottransfer usefullywithoutlearnedcalibration. Fixedconfidenceblend alsoworsened primaryMAE.',
 'Coverage andsourcecountalone cannotjustify replacing protectedCDF. No claimthat observedsegments are querygroundtruth.',
 'Nexttest learns source-to-query time transport andsourcebegin/endcontext fromTrain; no neighborcount/bandwidth sweep.'],
 preservation=dict(main_under131_06=True,nonparametric_sources_preserved=True,all_metrics_predictions_sources_retained=True),
 next_plan='plan_atencion_subtrayectos.json',Test='SEALED',goal='ACTIVE'))
p=BASE/'RESUMEN_RESULTADOS.md';s=p.read_text()
title='## Memoria de subtrayectos: combinación fija'
if title not in s:
 s+='\n\n'+title+'\n\nLa memoria directa obtuvo 163,945881 s; la combinación según evidencia obtuvo 130,959974 s en los mismos 127.941 viajes. Frente al mejor de 130,158042 s, la combinación empeoró 0,801932 s (IC exploratorio por día: [0,573310; 1,045564]). No pasó el criterio. Se guardan ambos resultados completos. La fórmula de 130,959974 s y sus dependencias quedan protegidas por estar por debajo de 131,06 s. Test cerrado.\n\nSe prepara una rama aprendida con fuentes Train excluidas por grupo semanal, contexto de inicio/final de cada fuente y seis épocas fijas. El ensayo previo no autoriza barrer pesos o radios en Validation.\n'
 p.write_text(s,encoding='utf-8')
state=read(OUT/'CONTINUIDAD_OPERATIVA.json');state['current_trial']=dict(experiment='atencion_subtrayectos_s0',stage='Buildingcompactcrossfit sourceindices; learned34Dsourceexpert sources prepared, not yettrained.',Test='sealed')
state['next_after_result']=['Finishcompactindexbank, Trainonly normalization, realGPUgradient/resourcepreflight, thenfixed6epochtraining.',
 'Inspectactualprocess/log/results, preserveallcompletedforecasts and qualifiedfullmodels/dependencies.',
 'If meaningfulgain, runmatched sourcevaluecontrol and freeze/reproducecandidate before any independentconfirmation.']
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
# Register every nonparametric qualified candidate as a source-complete dependency manifest.
write(OUT/'modelos_no_parametricos_protegidos.json',dict(candidates=[dict(id=r['id'],validation=r['validation'],
 source_bank='memoria_puntos_Train_s0/puntos_Train.npz',source_bank_sha256=r['source_bank_sha256'],
 required_parent=r['required_sources'][1],sources=r['sources'],protocol='memoria_subtrayectos_cdf_s0/protocolo.json',
 predictions_sha256=r['predictions_sha256'],reason='Userthreshold<131.06; not a newbest.')]))
print('Architecture, complete-result summary and conservation manifest updated.')
