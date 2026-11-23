from base_replanteo import *
runs=['mezcla_proyeccion_incierta_s0','control_dos_rutas_proyeccion_s0']
r,c=[read(OUT/x/'resultado.json') for x in runs]
certs=[read(OUT/x/'certificacion/resultado.json') for x in runs]
assert all(z['passed'] for z in certs)
assert r['initial_model_state_sha256']==c['initial_model_state_sha256']
a,b=[dict(np.load(OUT/x/'validacion.npz')) for x in runs];ref=dict(np.load(BEST/'validacion.npz'))
for key in ['row','timestamp','true']:assert np.array_equal(a[key],b[key]) and np.array_equal(a[key],ref[key])
result=dict(treatment=dict(id=runs[0],validation=r['validation'],best_epoch=r['best_epoch'],model_sha256=r['model_sha256']),
 control=dict(id=runs[1],validation=c['validation'],best_epoch=c['best_epoch'],model_sha256=c['model_sha256']),
 identical_initial_model_state=True,initial_state_sha256=r['initial_model_state_sha256'],
 fixed_epoch9_expanded_minus_control=paired(a['true'],a['epoch_9'],b['epoch_9'],a['timestamp']),
 fixed_epoch3_frozen_expanded_minus_control=paired(a['true'],a['epoch_3'],b['epoch_3'],a['timestamp']),
 selected_models_expanded_minus_control=paired(a['true'],a['pred'],b['pred'],a['timestamp']),
 selected_treatment_minus_original=paired(a['true'],a['pred'],ref['pred'],a['timestamp']),
 treatment_initial_untrained_validation=r['initial_validation'],
 full_prediction_reproduction_and_raw65_reverse_singletons_passed=True,
 checkpoints_all_preserved=[dict(id=x,saved_checkpoints=z['saved_checkpoints']) for x,z in zip(runs,[r,c])],
 threshold131_06_achieved=any(h['validation']['mae_s']<131.06 for z in [r,c] for h in z['all_epoch_validation']),
 confirmation_target73_5_achieved=False,Test_access=False,
 conclusion='Small exploratory improvement with8hypotheses; fixed-epoch paired controls support attribution under this single configuration. ReusedValidation and one seed, not confirmation, not external matched-cohort victory.')
write(OUT/'conclusion_mezcla_proyeccion.json',result)
best=dict(id=runs[0],epoch=r['best_epoch'],validation=r['validation'],model_file=r['model_file'],model_sha256=r['model_sha256'],
 certification=str((OUT/runs[0]/'certificacion/resultado.json').relative_to(ROOT)),scope='Verified fullValidation, exploratory, Testsealed',qualifies_below131_06=r['validation']['mae_s']<131.06,
 preserve_original_reference=dict(id=BEST.name,sha256=sha(BEST/'modelo.pt'),mae_s=read(BEST/'resultado.json')['validation']['mae_s']),
 all_completed_metrics_predictions_retained=True)
write(OUT/'nuevo_mejor_verificado.json',best)
state=read(OUT/'CONTINUIDAD_OPERATIVA.json')
state['retained_best']=best
state['protected_original_reference']=best['preserve_original_reference']
state['completed_projection_mixture']=dict(treatment=r['validation'],control=c['validation'],all9epochs_complete=True,certified=True,Test='sealed')
state['current_trial']=dict(experiment='historical_route_prototype_followup',stage='Train-only matched-count gate passed; prepare exact crossfit historical inputs next',no_GPU_job_active=True)
state['next_after_result']=['Historical prototype gate passed56.37percent geometry reduction at matched8candidate count onTrain20k, without querygeometry in selector. This is not MAE.','Develop historical six-prototype structured inputs with own-fold exclusion for every retrieved source and time statistic. Do not reuse neighbor cached movement profiles that include queryfold.','Use timestamps/Trainlabels only from eligible historical memory; querystillonlyfivefields. New calibrated time/geometry mixture should retain certified parent131.125167s.','One heavy GPU; all primary cohort/labels unchanged; Test sealed. Preserve all metrics/predictions and every qualified/newbest model.','Do not declare target achieved or close goal after a batch.']
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
# Extend the audit without mislabeling a new untrained mixture as learned improvement.
p=OUT/'consolidar_auditoria_resultados.py';code=p.read_text(encoding='utf-8-sig')
code=code.replace("def append(run,label,pred,reported,source,kind='selected_model',alpha=None):","def append(run,label,pred,reported,source,kind='selected_model',alpha=None,learned=True):")
old="new_learned_improvement=bool(not same and computed['mae_s']<reference_metric['mae_s']),"
new="new_function_improvement=bool(not same and computed['mae_s']<reference_metric['mae_s']),\\n new_learned_improvement=bool(learned and not same and computed['mae_s']<reference_metric['mae_s']),"
assert old in code;code=code.replace(old,new.replace('\\n','\n'))
old="pending=[run for run in ['campo_multiescala_s0','aux_subviajes_s0','atencion_vial_s0'] if not (OUT/run/'resultado.json').exists()]"
new="""for run,label in [('mezcla_proyeccion_incierta_s0','Mezcla con proyección incierta'),('control_dos_rutas_proyeccion_s0','Mezcla: control de dos rutas')]:
 source=OUT/run/'resultado.json'
 if source.exists():
  r=read(source);z=load_archive(source.parent/'validacion.npz')
  append(run,label+' — inicial nueva',z['epoch_0'],r['initial_validation'],source,kind='initial_candidate',learned=False)
  append(run,label+' — seleccionada',z['pred'],r['validation'],source,kind='selected_model',learned=r['learned_update_selected'])
  for row in r['all_epoch_validation']:
   epoch=row['epoch'];append(run,label+(' — final fija' if epoch==9 else ' — época '+str(epoch)),z['epoch_'+str(epoch)],row['validation'],source,kind='fixed_final_epoch' if epoch==9 else 'epoch_checkpoint')
pending=[run for run in ['campo_multiescala_s0','aux_subviajes_s0','atencion_vial_s0'] if not (OUT/run/'resultado.json').exists()]"""
assert old in code;code=code.replace(old,new);p.write_text(code,encoding='utf-8')
h=read(OUT/'piloto_prototipos_historicos_s0/resultado.json')
current=f"""<!-- estado_actual_verificado_inicio -->
Resultado vigente verificado: **MAE131,125167s**, mezcla con incertidumbre de proyección, época6. Referencia original131,256712s; control comparable de dos rutas131,248505s. Ambas reproducciones completas de127941predicciones y consultas crudas65pasaron. Mejora pequeña y exploratoria; todavía no se alcanza131,06s ni el objetivo73,5s. Test cerrado. Ver nuevo_mejor_verificado.json y conclusion_mezcla_proyeccion.json.

La siguiente vía tiene un diagnósticoTrain más fuerte: seis prototipos geométricos de otras semanas, junto con las dos rutas protegidas, bajaron el error oráculo de248,993851a108,638779m(56,37%) con el mismo número de ocho candidatos. No es un resultado deMAE. Se desarrollará una mezcla condicionada por esas rutas históricas; datos originales, mejores modelos y resultadosTransferTraj permanecen conservados.
<!-- estado_actual_verificado_fin -->

"""
for dest in [OUT/'README.md',BASE/'RESUMEN_RESULTADOS.md']:
 text=dest.read_text(encoding='utf-8-sig');assert '<!-- estado_actual_verificado_inicio -->' not in text;dest.write_text(current+text,encoding='utf-8')
# Recheck the published target only; do not equate its unavailable cohort with ours.
write(OUT/'reverificacion_transfertraj_20261122.json',dict(primary_source='https://papers.nips.cc/paper_files/paper/2025/file/9f02b7c937447b6857b21d31d5c344e6-Paper-Conference.pdf',
 checked=['Table3p8: PortoODTTE1.225±.006minutes,73.5seconds.','Table8p21:323481trips.','B.2: chronological8:1:1.','Task definitionp7: starting/ending locations and departure time.'],
 conclusion='Published target verified; exact processed cohort still unavailable, so no matched-cohort comparison claim.',primary_cohort_unchanged=True))
print(json.dumps(dict(treatment=r['validation'],control=c['validation'],fixed9_delta=result['fixed_epoch9_expanded_minus_control']['delta_mae_s'],
 selected_vs_original=result['selected_treatment_minus_original'],historical_gate=h['passed']),default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x))
