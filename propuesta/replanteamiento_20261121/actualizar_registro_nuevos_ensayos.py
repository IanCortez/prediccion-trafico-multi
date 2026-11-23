from base_replanteo import *
p=OUT/'consolidar_auditoria_resultados.py';s=p.read_text(encoding='utf-8-sig')
old="pending=[run for run in ['riesgo_total_s0','aux_solicitud_control_s0','aux_solicitud_semantic_s0'] if not (OUT/run/'resultado.json').exists()]"
new="""for run,label in [('aux_solicitud_control_s0_v2','Ajuste de control'),('aux_solicitud_semantic_s0_v2','Supervisión del tipo de solicitud'),('alumno_maestro_s0_v2','Alumno con maestro probabilístico')]:
 source=OUT/run/'resultado.json'
 if source.exists():
  r=read(source);z=load_archive(source.parent/'validacion.npz')
  append(run,label+' — inicial',z['epoch_0'],reference_metric,source,kind='initial_reference')
  for row in r['all_epoch_validation']:
   epoch=row['epoch'];append(run,label+('' if epoch==4 else ' — época '+str(epoch)),z['epoch_'+str(epoch)],row['validation'],source,kind='fixed_final_epoch' if epoch==4 else 'epoch_checkpoint')
source=OUT/'historicos_integrados_s0/resultado.json'
if source.exists():
 r=read(source);z=load_archive(source.parent/'validacion.npz')
 labels={'full_history_control':'Histórico completo: control','calendar_fold_control':'Histórico con exclusión semanal','integrated_five_views':'Integración de cinco históricos'}
 for name,item in r['outputs'].items():
  append(source.parent.name,labels[name],z[name],item['validation'],source,kind='matched_reference_control' if name=='full_history_control' else 'historical_policy')
pending=[run for run in ['historicos_integrados_s0'] if not (OUT/run/'resultado.json').exists()]"""
assert old in s;s=s.replace(old,new)
s=s.replace("['protected_reference','selected_model','supervised_type']","['protected_reference','selected_model','supervised_type','fixed_final_epoch','historical_policy','selected_risk_weight']")
s=s.replace("append(source.parent.name,'Riesgo de viaje completo, peso '+str(alpha),z['pred'][index],m,source,kind='declared_weight',alpha=alpha)","append(source.parent.name,'Riesgo de viaje completo, peso '+str(alpha),z['pred'][index],m,source,kind='selected_risk_weight' if alpha==r['selected_alpha'] else 'declared_weight',alpha=alpha)")
s=s.replace("fig,ax=plt.subplots(figsize=(11,8.6),dpi=160)","fig,ax=plt.subplots(figsize=(11,max(8.6,.45*len(chosen)+2.6)),dpi=160)")
compile(s,str(p),'exec');p.write_text(s,encoding='utf-8')
rcontrol=read(OUT/'aux_solicitud_control_s0_v2/resultado.json');rsemantic=read(OUT/'aux_solicitud_semantic_s0_v2/resultado.json');rstudent=read(OUT/'alumno_maestro_s0_v2/resultado.json')
assert all(not r['saved_checkpoints'] and r['best_epoch']==0 for r in [rcontrol,rsemantic,rstudent])
# Superseded teacher weights are no longer required by an active or queued student.
teacher=OUT/'maestro_cruzado_s0';meta=read(teacher/'resultado.json');removed=[]
for fold in meta['folds']:
 path=teacher/('maestro_fold_'+str(fold['fold'])+'.cbm');assert path.resolve().is_relative_to(OUT.resolve())
 if path.exists():
  assert sha(path)==fold['model_sha256'];removed.append(dict(file=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=fold['model_sha256']));path.unlink()
meta['teacher_weights_retained']=False;meta['cleanup_note']='Student completed without beating reference. All five OOF predicted parameter arrays remain consolidated, with exact source, metrics and original data; unused teacher fit weights removed.'
write(teacher/'resultado.json',meta);write(OUT/'limpieza_maestros_tras_alumno.json',dict(removed=removed,predicted_parameters_retained=True,params_sha256=meta['parameters_sha256'],original_reference_protected=True))
c=read(OUT/'CONTINUIDAD_OPERATIVA.json');c['auxiliary_queue']['complete']=True;c['soft_target_student_queue']['complete']=True;c['teacher_CPU']['weights_retained']=False
c['completed_auxiliary_studies']=dict(control_final_MAE_s=rcontrol['all_epoch_validation'][-1]['validation']['mae_s'],semantic_final_MAE_s=rsemantic['all_epoch_validation'][-1]['validation']['mae_s'],student_final_MAE_s=rstudent['all_epoch_validation'][-1]['validation']['mae_s'],all_selected_original_reference=True,new_candidates=[])
c['next_after_result']=['Current experiment is historicos_integrados_s0: frozen original model, profile integration and matched-construction control, no training. Read its actual process/log/result. Old controllers/teacher jobs are terminal.',
 'No auxiliary request or distillation candidate improved reference; each selected epoch0. Do not search more weights for the same trained outputs.',
 'Recompute all complete metrics from preserved predictions; every candidate below131.06 or any real new best must retain policy/model assets and pass raw reproduction. Test remains closed.',
 'If no improvement, draw the concrete conclusion and continue a genuinely distinct justified alternative. Goal confirmation73.5 remains unmet.']
write(OUT/'CONTINUIDAD_OPERATIVA.json',c)
print(json.dumps(dict(registry_source_updated=True,removed_teacher_models=len(removed),removed_MiB=sum(x['bytes'] for x in removed)/1024**2,goal='active')))
