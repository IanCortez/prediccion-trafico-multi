"""Register only a fully reloaded numerical minimum; no confirmationclaim."""
from base_replanteo import *
import shutil
run=OUT/'atencion_subtrayectos_s0';r=read(run/'resultado.json');proof=read(run/'reproduccion_independiente.json')
assert r['complete'] and r['qualified'] and proof['passed'] and proof['all127941quantiles_exact'] and proof['raw65_reverse_singleton_exact']
assert proof['reloaded_model_sha256']==r['model_sha256'] and sha(r['model_file'])==r['model_sha256']
current=read(OUT/'nuevo_mejor_verificado.json')
assert current['id']==r['config']['parent_id'] and r['validation']['mae_s']<current['validation']['mae_s']
assert current['model_sha256']==r['config']['parent_sha256']
original=OUT/'mejor_previo_atencion_subtrayectos.json';assert not original.exists();write(original,current)
artifact=run/'reproduccion_fuentes';artifact.mkdir()
for name in ['inferencia_atencion_subtrayectos.py','reproducir_atencion_subtrayectos.py','recuperar_evaluacion_atencion_subtrayectos.py']:
 shutil.copy2(OUT/name,artifact/name)
best=dict(id=r['id'],epoch=6,validation=r['validation'],model_file=r['model_file'],model_sha256=r['model_sha256'],
 checkpoint_kind='Branchonly34Dsubtrajectoryattention delta; immutableprotected30componentparent required.',
 required_parent={key:current[key] for key in ['id','model_file','model_sha256']},
 ancestor_chain=current['required_parent'],certification=str((run/'reproduccion_independiente.json').relative_to(ROOT)),
 scope='Reloadedfinalcheckpoint full127941quantiles and raw65fivefields/reverse/singleton exact. Fixed6epochs; one seed/reusedValidation, not confirmation.',
 qualifies_below131_06=True,preserve_original_reference=current['preserve_original_reference'],
 historical_memory_sha256=current['historical_memory_sha256'],subtrajectory_point_memory_sha256=r['config']['source_memory_sha256'],
 source_indices_manifest_sha256=r['source_manifest_sha256'],normalization_sha256=r['normalization_sha256'],
 paired_vs_parent=r['paired_vs_parent'],all_completed_metrics_predictions_retained=True,
 control='Atleast.5s improvement gate failed despite favorableexploratorydayCI; no matchedcontrol or causalclaim.',
 practical_gain_established=False,uncertainty=r['uncertainty'],reproduce='reproducir_atencion_subtrayectos.py',
 reproduction_sources={p.name:sha(p) for p in artifact.glob('*.py')})
write(OUT/'nuevo_mejor_verificado.json',best)
lineage=read(OUT/'modelos_mejores_protegidos.json');assert not any(v['id']==r['id'] for v in lineage)
lineage.append(best);write(OUT/'modelos_mejores_protegidos.json',lineage)
dependencies={'memoria_puntos_Train_s0/puntos_Train.npz':r['config']['source_memory_sha256'],
 'indices_subtrayectos_s0/train_indices.npz':read(OUT/'indices_subtrayectos_s0/resultado.json')['outputs']['train']['sha256'],
 'indices_subtrayectos_s0/validation_indices.npz':read(OUT/'indices_subtrayectos_s0/resultado.json')['outputs']['validation']['sha256'],
 'atencion_subtrayectos_s0/constructor_tokens.npz':r['normalization_sha256'],
 str(Path(current['model_file']).relative_to(OUT)):current['model_sha256']}
write(OUT/'modelos_atencion_subtrayectos_protegidos.json',dict(candidates=[dict(id=r['id'],validation=r['validation'],
 model_file=r['model_file'],model_sha256=r['model_sha256'],dependencies=dependencies,qualified=True)]))
state=read(OUT/'CONTINUIDAD_OPERATIVA.json');state['retained_best']=best
state['goal_status']='ACTIVE; verified numericalValidation minimum129.990034s. Small.168007s gain, meaningful.5sgatefailed;73.5confirmation unmet,Testsealed.'
state['current_trial']=dict(experiment='subtrayectos_conjuntos_s0',stage='Sources reviewed, previousfrozenbranch fullyreproduced; jointfreshtrainingnext.',Test='sealed')
state['next_after_result']=['Runreviewedfreshjoint24epochmodel afterclosedpriorprocess; no pretrainedneuralweights.',
 'ActualTrain gradients and canonical2048inferenceresourceproof beforeupdates; releaseTrain arraysbeforefinalValidation.',
 'Ifmeaningfulgain, matched sourcevaluecontrol andfreeze/reproducebefore anyindependentconfirmation.',
 'Preserve41qualifiedneuralcheckpoints, numericalbestlineage, complete nonparametric130.960formula andallpredictions including>131.06.']
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
write(OUT/'arquitectura_activa.json',dict(status_source='CONTINUIDAD_OPERATIVA.json',protected_best_source='nuevo_mejor_verificado.json',
 architecture_document='propuesta/ARQUITECTURA.md',current_experiment='subtrayectos_conjuntos_s0',
 current_protocol='plan_subtrayectos_conjuntos.json',protected_best=r['validation'],public_query=r['config']['public_query'],Test='sealed',goal='ACTIVE'))
p=OUT/'consolidar_auditoria_resultados.py';s=p.read_text();marker='# Learned ordered-subtrajectory expert, fixed six epochs.'
if marker not in s:
 insert="""
# Learned ordered-subtrajectory expert, fixed six epochs.
source=OUT/'atencion_subtrayectos_s0/resultado.json'
if source.exists():
 r=read(source);z=load_archive(source.parent/'validacion.npz')
 append(source.parent.name,'Atención sobre subtrayectos: final fija',z['pred'],r['validation'],source)
source=OUT/'subtrayectos_conjuntos_s0/resultado.json'
if source.exists():
 r=read(source);z=load_archive(source.parent/'validacion.npz')
 append(source.parent.name,'Subtrayectos conjuntos desde cero',z['pred'],r['validation'],source)
"""
 s=s.replace('pending=[',insert+'\npending=[');p.write_text(s)
p=BASE/'RESUMEN_RESULTADOS.md';s=p.read_text()
s+='\n\n## Atención aprendida sobre subtrayectos: 129,990034 s\n\nTras seis épocas fijas con todos los viajes Train, el MAE bajó de 130,158042 a 129,990034 s: diferencia −0,168007 s, intervalo exploratorio por día [−0,222310; −0,110332]. RMSE 209,651248 s, mediana del error absoluto 75 s. La mejora en MAE es pequeña; RMSE empeora ligeramente y no se cumple la puerta de mejora práctica de 0,5 s. Se conserva como mínimo numérico verificado, sin afirmar confirmación ni ventaja causal del mecanismo.\n\nLos pesos guardados se recargaron y reprodujeron todos los cuantiles de las 127.941 consultas, más 65 consultas crudas con inversión de orden y caso individual. La evaluación final se recuperó en un proceso que carga solo Validation tras superar el límite de memoria al conservar Train residente. Se verificó que los pesos coinciden con la recuperación de la época 6. No se repitió el entrenamiento.\n\nEl modelo, de 0,90 MiB como rama adicional, necesita el padre de atención histórica y su cadena protegida, la memoria de puntos y los índices compactos. Receta: replanteamiento_20261121/reproducir_atencion_subtrayectos.py. Test permanece cerrado.\n'
p.write_text(s,encoding='utf-8')
write(OUT/'conclusion_atencion_subtrayectos.json',dict(complete=True,validation=best['validation'],
 paired_vs_parent=best['paired_vs_parent'],new_verified_numerical_best=True,practical_gate=False,
 warning='Exploratory adaptiveValidation; RMSEslightlyworse; no independentconfirmation or causalclaim.',
 next='Jointfreshinitialization trainsroot+sources together, ratherthan extending samefrozenbranch or tuning sixepochs.',
 Test='sealed',goal='ACTIVE'))
print(json.dumps(dict(best=best['validation'],all_predictions_and41qualifiedweights_preserved=True,goal='ACTIVE')))
