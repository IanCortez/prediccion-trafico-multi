from base_replanteo import *
run=OUT/'atencion_vial_s0';r=read(run/'resultado.json');z=dict(np.load(run/'validacion.npz'));ref=dict(np.load(BEST/'validacion.npz'))
for key in ['row','timestamp','true']:assert np.array_equal(z[key],ref[key])
assert len(r['all_epoch_validation'])==9 and len(r['saved_checkpoints'])==0 and r['best_epoch']==0
for row in r['all_epoch_validation']:assert metric(z['true'],z['epoch_'+str(row['epoch'])])==row['validation']
best=min(r['all_epoch_validation'],key=lambda x:x['validation']['mae_s'])
result=dict(id=r['id'],best_learned_epoch=best,protected_reference=metric(ref['true'],ref['pred']),
 final_fixed_epoch_vs_reference=paired(z['true'],z['epoch_9'],ref['pred'],ref['timestamp']),
 all_nine_epoch_metrics_and_predictions_preserved=True,selected_epoch0_is_original_reference=True,
 conditional_information_control='Skipped per predeclared rule; no learned epoch improved protected reference.',
 conclusion='This public-road attention configuration did not improve primary MAE. Later joint training reduced Train loss while validation worsened; no universal claim about attention models.',
 Test_access=False,goal='ACTIVE; target not achieved')
write(OUT/'conclusion_atencion_vial.json',result)
# Add only the finished new trial; never count geometric pilot scores as primary MAE.
p=OUT/'consolidar_auditoria_resultados.py';code=p.read_text(encoding='utf-8-sig')
old="pending=[run for run in ['campo_multiescala_s0','aux_subviajes_s0'] if not (OUT/run/'resultado.json').exists()]"
new="""for run,label in [('atencion_vial_s0','Atención sobre rutas públicas'),('control_reloj_atencion_vial_s0','Atención: control sin información vial')]:
 source=OUT/run/'resultado.json'
 if source.exists():
  r=read(source);z=load_archive(source.parent/'validacion.npz')
  append(run,label+' — inicial',z['epoch_0'],reference_metric,source,kind='initial_reference')
  for row in r['all_epoch_validation']:
   epoch=row['epoch'];append(run,label+('' if epoch==9 else ' — época '+str(epoch)),z['epoch_'+str(epoch)],row['validation'],source,kind='fixed_final_epoch' if epoch==9 else 'epoch_checkpoint')
pending=[run for run in ['campo_multiescala_s0','aux_subviajes_s0','atencion_vial_s0'] if not (OUT/run/'resultado.json').exists()]"""
assert old in code;p.write_text(code.replace(old,new),encoding='utf-8')
deleted=[];data=OUT/'secuencias_viales_od_s0';record=read(data/'resultado.json')
for name in ['train_secuencias.npz','validation_secuencias.npz']:
 path=data/name;assert path.resolve().is_relative_to(OUT.resolve())
 assert path.stat().st_size==record['outputs'][name]['bytes'] and sha(path)==record['outputs'][name]['sha256']
 deleted.append(dict(file=name,bytes=path.stat().st_size,sha256=sha(path)));path.unlink()
write(data/'limpieza_cache_descartado.json',dict(files=deleted,bytes=sum(x['bytes'] for x in deleted),reason='Finished nine-epoch configuration did not beat reference.',
 exact_regeneration_script='reconstruir_secuencias_viales_od.py',every_completed_metric_prediction_source_preserved=True,
 confidence_counts_retained_sha256=sha(data/'frecuencias_Train.npy'),protected_model_unchanged=sha(BEST/'modelo.pt')==r['config']['parent_sha256']))
text=f"""

<!-- cierre_atencion_y_diagnostico_proyeccion -->
La atención sobre dos rutas públicas terminó las nueve épocas previstas: mejor época aprendida4, MAE{best['validation']['mae_s']:.6f}s, frente a131,256712s. La época9 obtuvo132,545431s; su empeoramiento frente a la referencia es{result['final_fixed_epoch_vs_reference']['delta_mae_s']:.6f}s. La época0 seleccionada es el modelo original; no se presenta como ganancia aprendida. El descenso de pérdida Train durante las épocas conjuntas no se trasladó a validación. No se ejecutó la ablación condicional porque no hubo mejora.

Se conservaron las nueve predicciones y métricas, con todos sus viajes y etiquetas comprobados. Se retiraron{sum(x['bytes'] for x in deleted):,}bytes de cachés de rutas descartados. Se mantienen fuentes, frecuencias ajustadas solo enTrain, manifiestos y el instrumento reconstruir_secuencias_viales_od.py, que regenera los cachés desde las consultas originales y comprueba sus hashes.

Antes de otro modelo GPU se medirá en20000viajes deTrain si varias conexiones plausibles de los extremos al mapa añaden cobertura útil. Se comparan las dos rutas protegidas y una recta contra esas mismas opciones más seis rutas por otras tres parejas de proyecciones. Las nuevas parejas se eligen por distancia GPS, sin etiquetas. La geometría real se usa solo para medir un selector oráculo: no es una predicción OD desplegable ni un resultado deMAE. Umbral fijado antes del piloto: reducción media geométrica de al menos2% y5m. Un fallo de compilación deNumba anterior a cualquier resultado se archivó y se corrigió sin cambiar datos, matemáticas ni umbral.

El objetivo73,5s sigue pendiente. Se conservan métricas/predicciones incluso cuando superan131,06s, el mejor modelo y los resultadosTransferTraj; Test cerrado.
"""
for dest in [OUT/'README.md',BASE/'RESUMEN_RESULTADOS.md']:
 t=dest.read_text(encoding='utf-8-sig');assert '<!-- cierre_atencion_y_diagnostico_proyeccion -->' not in t;dest.write_text(t+text,encoding='utf-8')
state=read(OUT/'CONTINUIDAD_OPERATIVA.json');state['previous_goal_turn_classification']='PROGRESS: completed and audited auxiliary experiment; built and ran new public-road representation; no target claim.'
state['completed_attention']=dict(id='atencion_vial_s0',epochs=9,best_learned_mae_s=best['validation']['mae_s'],best_epoch_selected=0,learned_improvement=False,cache_removed_bytes=sum(x['bytes'] for x in deleted))
state['current_trial']=dict(experiment='piloto_proyeccion_incierta_s0_v2',scope='Train-only geometric coverage gate, not primary MAE',session=22022,Test='sealed')
state['next_after_result']=['Check actual pilot process/session and its Train-only result; no blind relaunch on timeout.','If the fixed geometric coverage gate passes, formulate the uncertainty-aware public route mixture and execute a new complete primary trial.','If it fails, do not enlarge that bank or tune the gate from validation; pursue a distinct mechanism with a stronger rationale.','Do not edit active or queued source files. Keep all completed metrics/predictions, original best, qualifying candidates, TransferTraj results.','Goal still unmet; Test remains sealed.']
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
print(json.dumps(dict(attention_complete=True,best_learned_mae_s=best['validation']['mae_s'],learned_improvement=False,removed_cache_MiB=sum(x['bytes'] for x in deleted)/1024**2)))
