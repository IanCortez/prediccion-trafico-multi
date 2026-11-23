"""Recompute all completed full-cohort outputs from preserved prediction archives."""
from base_replanteo import *
import importlib.util
access=deny_test();start=time.time();reference=dict(np.load(BEST/'validacion.npz'))
assert len(reference['true'])==127941
reference_metric=metric(reference['true'],reference['pred'])
rows=[];archives=[]
def append(run,label,pred,reported,source,kind='selected_model',alpha=None,learned=True):
 assert reported['n']==127941
 computed=metric(reference['true'],pred)
 errors={k:abs(computed[k]-reported[k]) for k in ['mae_s','rmse_s','mdae_s']}
 assert max(errors.values())<1e-5,(run,label,errors)
 same=np.array_equal(pred,reference['pred'])
 rows.append(dict(run=run,label=label,kind=kind,alpha=alpha,computed_validation=computed,
 reported_metric_max_error=max(errors.values()),source=str(source.relative_to(ROOT)),
 equals_reference_predictions=bool(same),new_function_improvement=bool(not same and computed['mae_s']<reference_metric['mae_s']),
 new_learned_improvement=bool(learned and not same and computed['mae_s']<reference_metric['mae_s']),
 below_131_06_s=bool(computed['mae_s']<131.06),below_external_73_5_s=bool(computed['mae_s']<73.5)))
def load_archive(path):
 z=dict(np.load(path))
 for k in ['row','timestamp','true']:assert np.array_equal(z[k],reference[k]),(path,k,'Cohort/label mismatch')
 archives.append(dict(file=str(path.relative_to(ROOT)),sha256=sha(path),same_rows_labels_and_timestamps=True))
 return z
append(BEST.name,'Referencia conservada',reference['pred'],reference_metric,BEST/'resultado.json',kind='protected_reference')
main=[
 ('descompuesto_aux0_s0','Movimiento y permanencia: control'),
 ('descompuesto_aux1_s0','Movimiento y permanencia: supervisión'),
 ('memoria_cdf_s0','Memoria de distribuciones'),
 ('memoria_mse_s0','Memoria con error cuadrático'),
 ('base_oof_estatica_s0','Base fuera de bloque'),
 ('memoria_residual_oof_s0','Memoria de residuos fuera de bloque'),
 ('espacial_semimarkov_s0_v2','Proceso espacial con permanencias'),
 ('espacial_direccional_s0','Proceso espacial con dirección'),
 ('espacial_prior_cf_s0','Dirección y conteos históricos'),
 ('espacial_fases_referencia_s0','Permanencias por fases')]
for run,label in main:
 source=OUT/run/'resultado.json';r=read(source);z=load_archive(OUT/run/'validacion.npz')
 append(run,label,z['pred'],r['validation'],source)
 if 'graph_validation' in r:
  key='graph_with_reference_fallback' if 'graph_with_reference_fallback' in z else 'graph_with_support_fallback'
  append(run,label+' — rama espacial',z[key],r['graph_validation'],source,kind='secondary_branch')
source=OUT/'prefijo_nativo_s0/resultado.json';r=read(source);z=load_archive(source.parent/'validacion.npz')
assert z['pred'].shape==(5,127941)
for index,(alpha,m) in enumerate(zip(z['alphas'],r['all_alpha_validation'])):
 append(source.parent.name,'Generación nativa, peso '+str(float(alpha)),z['pred'][index],m,source,kind='declared_weight',alpha=float(alpha))
source=OUT/'calibracion_unica_fases_s0/resultado.json';r=read(source);z=load_archive(source.parent/'predicciones_distintas.npz')
for alpha,index,m in zip(z['alphas'],z['alpha_to_row'],r['full_validation']):
 pred=reference['pred'] if index<0 else z['pred'][index]
 append(source.parent.name,'Calibración fija, peso '+str(float(alpha)),pred,m,source,kind='declared_weight',alpha=float(alpha))
source=OUT/'tipo_latente_cpu_s0/resultado.json';r=read(source);z=load_archive(source.parent/'validacion.npz')
for name,label in [('matched_unsupervised','Corrección de distribución: control'),('supervised_type','Corrección con tipo de solicitud latente')]:
 append(source.parent.name,label,z[name],r['outputs'][name]['validation'],source,kind=name)
# Include the currently running trial only once its complete result is actually available.
source=OUT/'riesgo_total_s0/resultado.json'
if source.exists():
 r=read(source);z=load_archive(source.parent/'validacion.npz')
 for index,(alpha,m) in enumerate(zip(r['config']['alphas'],r['all_alpha_validation'])):
  append(source.parent.name,'Riesgo de viaje completo, peso '+str(alpha),z['pred'][index],m,source,kind='selected_risk_weight' if alpha==r['selected_alpha'] else 'declared_weight',alpha=alpha)
for run,label in [('aux_solicitud_control_s0_v2','Ajuste de control'),('aux_solicitud_semantic_s0_v2','Supervisión del tipo de solicitud'),('alumno_maestro_s0_v2','Alumno con maestro probabilístico')]:
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
for run,label in [('campo_multiescala_s0','Campo espacial desde cero'),('control_capacidad_campo_s0','Control de capacidad del campo')]:
 source=OUT/run/'resultado.json'
 if source.exists():
  r=read(source);z=load_archive(source.parent/'validacion.npz')
  append(run,label+' — mejor época',z['pred'],r['validation'],source,kind='fresh_selected')
  for row in r['all_epoch_validation']:
   epoch=row['epoch'];append(run,label+' — época '+str(epoch),z['epoch_'+str(epoch)],row['validation'],source,kind='epoch_checkpoint')
for run,label in [('aux_subviajes_s0','Supervisión auxiliar de subtrayectos'),('control_aux_subviajes_s0','Control auxiliar de subtrayectos')]:
 source=OUT/run/'resultado.json'
 if source.exists():
  r=read(source);z=load_archive(source.parent/'validacion.npz')
  append(run,label+' — inicial',z['epoch_0'],reference_metric,source,kind='initial_reference')
  for row in r['all_epoch_validation']:
   epoch=row['epoch'];append(run,label+('' if epoch==6 else ' — época '+str(epoch)),z['epoch_'+str(epoch)],row['validation'],source,kind='fixed_final_epoch' if epoch==6 else 'epoch_checkpoint')
for run,label in [('atencion_vial_s0','Atención sobre rutas públicas'),('control_reloj_atencion_vial_s0','Atención: control sin información vial')]:
 source=OUT/run/'resultado.json'
 if source.exists():
  r=read(source);z=load_archive(source.parent/'validacion.npz')
  append(run,label+' — inicial',z['epoch_0'],reference_metric,source,kind='initial_reference')
  for row in r['all_epoch_validation']:
   epoch=row['epoch'];append(run,label+('' if epoch==9 else ' — época '+str(epoch)),z['epoch_'+str(epoch)],row['validation'],source,kind='fixed_final_epoch' if epoch==9 else 'epoch_checkpoint')
for run,label in [('mezcla_proyeccion_incierta_s0','Mezcla con proyección incierta'),('control_dos_rutas_proyeccion_s0','Mezcla: control de dos rutas')]:
 source=OUT/run/'resultado.json'
 if source.exists():
  r=read(source);z=load_archive(source.parent/'validacion.npz')
  append(run,label+' — inicial nueva',z['epoch_0'],r['initial_validation'],source,kind='initial_candidate',learned=False)
  append(run,label+' — seleccionada',z['pred'],r['validation'],source,kind='selected_model',learned=r['learned_update_selected'])
  for row in r['all_epoch_validation']:
   epoch=row['epoch'];append(run,label+(' — final fija' if epoch==9 else ' — época '+str(epoch)),z['epoch_'+str(epoch)],row['validation'],source,kind='fixed_final_epoch' if epoch==9 else 'epoch_checkpoint')
pending=[run for run in ['campo_multiescala_s0','aux_subviajes_s0','atencion_vial_s0'] if not (OUT/run/'resultado.json').exists()]
result=dict(scope='Completed replanteamiento_20261121 full unchanged Validation outputs plus protected reference; not all historical thesis runs',
 goal_confirmation_s=73.5,preservation_threshold_s=131.06,reference=reference_metric,
 full_prediction_outputs=len(rows),distinct_archives_checked=len(archives),experiments=rows,archives=archives,
 pending_no_metric_claimed=pending,
 exact_reference_copies_are_not_learned_gains=True,
 Test_files_accessed=False,actual_data_files_accessed=sorted(access),
 limitations=['Repeated adaptive Validation and exploratory selection; not confirmation.','TransferTraj published cohort differs;73.5s remains an external target, not a matched-cohort victory.','No universal irreducible-error claim follows from these finite candidate outputs.'],
 code_sha256=sha(__file__),elapsed_seconds=time.time()-start)
write(OUT/'registro_consolidado_auditado.json',result)
# A compact scientific figure for the result summary; no interactive or external dependency.
if importlib.util.find_spec('matplotlib'):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 chosen=[q for q in rows if q['kind'] in ['protected_reference','selected_model','supervised_type','fixed_final_epoch','historical_policy','selected_risk_weight','fresh_selected']]
 native=[q for q in rows if q['run']=='prefijo_nativo_s0' and q['alpha']==1][0]
 chosen.append(native);chosen.sort(key=lambda q:q['computed_validation']['mae_s'])
 fig,ax=plt.subplots(figsize=(11,max(8.6,.45*len(chosen)+2.6)),dpi=160)
 pos=np.arange(len(chosen));values=[q['computed_validation']['mae_s'] for q in chosen]
 colors=['#276a53' if q['kind']=='protected_reference' else '#6c849a' for q in chosen]
 ax.barh(pos,values,color=colors,height=.65)
 ax.set_yticks(pos,[q['label'].replace('Generación nativa, peso 1.0','Generación nativa: peso máximo') for q in chosen],fontsize=10)
 ax.invert_yaxis();ax.set_ylim(len(chosen)+.5,-.5);ax.set_xlim(0,165);ax.set_xlabel('MAE en segundos · menor es mejor',fontsize=11)
 ax.axvline(73.5,color='#9d493d',linestyle='--',linewidth=1.5,label='Meta externa: 73,5 s')
 ax.axvline(reference_metric['mae_s'],color='#276a53',linestyle=':',linewidth=1.3,label='Referencia: 131,26 s')
 for y,value in zip(pos,values):ax.text(value+1,y,f'{value:.2f}'.replace('.',','),va='center',fontsize=9)
 ax.set_title('Validación completa · los mismos 127.941 viajes',loc='left',fontsize=15,pad=17,fontweight='bold')
 ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True)
 ax.legend(loc='lower left',frameon=False,fontsize=9)
 fig.text(.02,.035,'Métricas recalculadas desde predicciones guardadas. Búsqueda adaptativa: resultados exploratorios.\nLa cohorte publicada de TransferTraj es distinta. El generador eligió peso 0 (referencia); aquí se muestra también su peso máximo.\nLos ensayos todavía activos no se incluyen como resultados. Test permanece cerrado.',fontsize=8,color='#454545')
 fig.subplots_adjust(left=.40,right=.96,top=.90,bottom=.16)
 fig.savefig(OUT/'comparacion_resultados_auditados.png',facecolor='white')
 plt.close(fig)
print(json.dumps(dict(outputs=len(rows),archives=len(archives),all_rows_labels_and_timestamps_match=True,maximum_reported_metric_error=max(q['reported_metric_max_error'] for q in rows),improved_run_ids=sorted(set(q['run'] for q in rows if q['new_learned_improvement'])),pending=pending,seconds=time.time()-start)))
