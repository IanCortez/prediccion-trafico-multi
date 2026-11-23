from base_replanteo import *
import shutil
# Both processes are terminal; scientific files were written before console serialization failed.
events=[]
for name,old,new in [
 ('ejecutar_aux_subviajes.py','print(json.dumps(conclusion),flush=True)',"print(json.dumps(dict(stage='complete',learned_improvement=conclusion['learned_improvement'],seconds=conclusion['seconds'])),flush=True)"),
 ('preparar_secuencias_viales_od.py','print(json.dumps(result),flush=True)',"print(json.dumps(dict(stage='complete',cache_bytes=cache_bytes,seconds=result['seconds'],Test_access=False)),flush=True)")]:
 p=OUT/name;code=p.read_text(encoding='utf-8-sig');assert old in code
 archive=OUT/(name+'.impresion_v1.txt');assert not archive.exists();shutil.copy2(p,archive)
 p.write_text(code.replace(old,new),encoding='utf-8')
 events.append(dict(file=name,executed_source_archive=archive.name,original_source_sha256=sha(archive),fix='Console summary only; NumPy arrays already serialized correctly in retained JSON files. No data/model rerun.'))
write(OUT/'correccion_impresion_tras_resultados_completos.json',events)
r=read(OUT/'aux_subviajes_s0/resultado.json');c=read(OUT/'conclusion_aux_subviajes.json')
s=read(OUT/'secuencias_viales_od_s0/resultado.json');g=read(OUT/'secuencias_viales_od_s0/verificacion_entrada.json')
assert g['passed'] and not s['Validation_labels_accessed'] and not s['Test_access']
text=f"""

<!-- cierre_subviajes_y_atencion_vial -->
El ensayo auxiliar de subtrayectos concluyó las seis épocas. Los MAE aprendidos fueron131,370085;131,369967;131,369264;131,397519;131,359885;131,362933s. Ninguno mejora131,256712s; la época0 seleccionada sigue siendo la referencia, sin avance aprendido. La época6 empeora en0,106221s, IC95% exploratorio por día[0,004467;0,212297]. El control condicional no se ejecutó. Se mantienen las seis predicciones y métricas, las consultas/etiquetas/procedencia auxiliares y sus fuentes. Se retiró únicamente el caché de características206.299.998bytes; reconstruir_features_subviajes.py permite regenerarlo.

La siguiente alternativa aporta composición vial ordenada: dos rutas generadas con el mapa público2014, elegidas exactamente como las rutas longitud/tiempo libre del modelo protegido, se resumen con32identificadores de tramo equidistantes por ruta. La consulta conserva solo origen,destino,fecha. No recibe trayectoria observada, cantidad de puntos ni duración. Se mantuvieron todos los viajes de Train y validación; la preparación no leyó etiquetas de validación. La comprobación de65consultas de Train recuperó longitudes y tiempos libres con error máximo{g['max_error_m_or_s']:.8f}, y pasó orden inverso, consultas individuales y rechazo de trayectoria adicional. El caché comprimido ocupa{s['cache_bytes']/1024**2:.2f}MiB.

Se evaluará atención sobre estos tramos: embeddings16dim, seis variables geométricas de mapa y ocho de calendario; Transformer64dim,dos capas,cuatro cabezas, y corrección de24parámetros de tiempo de los ocho componentes. La salida correctora empieza en cero. Se fijan tres épocas con la torre global congelada y seis conjuntas. Es una arquitectura local inspirada en el modelado de composición y secuencias, no una réplica de UNITE. Solo una mejora real justifica ejecutar la ablación que elimina la información de los tramos y conserva calendario/código; no se la presentará como igualdad de capacidad efectiva.

Todas las métricas y predicciones se conservan, incluso por encima de131,06s. El objetivo73,5s sigue pendiente; Test permanece cerrado.
"""
for p in [OUT/'README.md',BASE/'RESUMEN_RESULTADOS.md']:
 old=p.read_text(encoding='utf-8-sig');assert '<!-- cierre_subviajes_y_atencion_vial -->' not in old;p.write_text(old+text,encoding='utf-8')
review=dict(date='2026-11-22',primary_sources=[
 dict(title='TrajCogn,IJCAI2025',url='https://ijcai.org/proceedings/2025/411',
      verified='Sections3.2,4.4,5.1,5.4: trajectory learning; PortoTTE explicitly excluded due equal-interval sampling leakage; its51.973sTTE is Chengdu, not PortoOD.',
      decision='Do not treat it as PortoOD benchmark or expose query sample count.'),
 dict(title='Efficient and representative door-to-door travel time estimation,2026',url='https://link.springer.com/article/10.1007/s11116-025-10705-7',
      verified='Switzerland travel surveys and routing/built-environment correction; intersection count/density along route predictive.',
      inference='Detailed public-route composition is a testable input refinement; Swiss performance does not establish Porto accuracy.'),
 dict(title='UNITE,KDD2026',url='https://web.comp.polyu.edu.hk/csmlyiu/conf/KDD26_UNITE.pdf',
      verified='Progress-state route/time modeling, movement histograms and local traffic. China cities, not Porto.',
      local_source_inspection='models/tte.py: teacher_forcing_ratio=-1 uses fixed50step generation; ratio0 still masks using observed lengths. Any official adaptation must use fully autonomous query-only inference.',
      decision='Do not execute official defaults or claim local attention branch is a replication.'),
 dict(title='STFD-ETA,publisher result found',url='https://www.sciencedirect.com/science/article/pii/S0950705126019866',
      status='Search excerpt only; full publisher access403; issue date4Dec2026 after current22Nov. OD input/protocol not verified.',
      decision='No matched benchmark claim or adoption based on headline MAE.')
],files_changed_outside_project=False)
write(OUT/'revision_composicion_vial_20261122.json',review)
state=read(OUT/'CONTINUIDAD_OPERATIVA.json')
state['completed_subtrip_auxiliary']=dict(id=r['id'],best_learned_mae_s=131.35988463432363,selected_epoch=0,learned_improvement=False,cache_removed_bytes=206299998,all_metrics_predictions_retained=True)
state['current_trial']=dict(experiment='atencion_vial_s0',stage='Ready to launch fixed3frozen+6joint epochs; public route preparation complete',sources_frozen_on_launch=True)
state['public_route_data']=dict(id='secuencias_viales_od_s0',cache_bytes=s['cache_bytes'],gate_passed=True,Validation_labels_used_in_preparation=False)
state['next_after_result']=['One GPU; inspect actual atencion_vial process/session and history. Never edit active/queued model/trainer sources.','Freeze any actual improvement, preserve full metrics/predictions at all MAE levels.','Only on learned gain run separate clock-only route-information ablation; not equal effective capacity.','If target candidate appears, raw5field reconstruction/permutation/singleton checks before any Test.','If negative, remove exact discarded route cache after retaining sources, metrics,predictions and regeneration recipe; continue a genuinely distinct justified improvement.','Goal73.5 still unmet; Test remains sealed.']
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
print(json.dumps(dict(public_route_cache_MiB=s['cache_bytes']/1024**2,public_gate=g['passed'],subtrip_experiment_complete=True)))
