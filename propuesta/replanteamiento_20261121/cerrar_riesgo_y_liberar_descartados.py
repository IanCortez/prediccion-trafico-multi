from base_replanteo import *
rpath=OUT/'riesgo_total_s0/resultado.json';r=read(rpath);z=dict(np.load(rpath.parent/'validacion.npz'));ref=dict(np.load(BEST/'validacion.npz'))
for k in ['row','timestamp','true']:assert np.array_equal(z[k],ref[k])
assert not r['qualified_candidates'] and all(m['n']==127941 and m['mae_s']>=131.25671207822356 for m in r['all_alpha_validation'])
assert np.array_equal(z['pred'][0],ref['pred'])
outputs=[]
for alpha,pred,m in zip(z['alphas'],z['pred'],r['all_alpha_validation']):
 actual=metric(z['true'],pred);assert abs(actual['mae_s']-m['mae_s'])<1e-10
 outputs.append(dict(alpha=float(alpha),validation=actual,paired_vs_reference=paired(z['true'],pred,ref['pred'],z['timestamp'])))
write(OUT/'conclusion_riesgo_completa.json',dict(outputs=outputs,selected_alpha=r['selected_alpha'],steps_run=r['steps_run'],best_step=r['best_step'],support_failure_rate=r['support_failure_rate'],reference_predictions_exact=True,predictions_sha256=sha(rpath.parent/'validacion.npz'),conclusion='Bounded full-trip CRPS training did not improve the unchanged full-cohort reference at any declared weight. The512 selection improvement did not generalize. Do not extend this identical configuration merely to chase the threshold.',limitations='Finite trial, not proof that all generative models fail. Native and risk used different canonical batch sizes, so cross-run branch differences are not a clean causal estimate.',Test='sealed',goal='active'))
# Only explicit failed/finished weight files; no original data, live model or teacher assets.
targets=[
 OUT/'riesgo_total_s0/modelo.pt',OUT/'riesgo_total_s0/ultima_reanudacion.pt',
 OUT/'prefijo_nativo_s0/modelo.pt',OUT/'espacial_prior_cf_s0/modelo.pt',
 OUT/'espacial_fases_referencia_s0/modelo.pt',
 OUT/'piloto_maestro_probabilistico_s0/OD_only.cbm',OUT/'piloto_maestro_probabilistico_s0/privileged_geometry.cbm']
removed=[]
for p in targets:
 assert p.resolve().is_relative_to(OUT.resolve())
 if p.exists():
  size=p.stat().st_size;assert size<50*1024**2
  removed.append(dict(file=str(p.relative_to(ROOT)),bytes=size,sha256=sha(p),reason='Completed discarded lineage or superseded pilot; no current/queued trainer depends on these weights'))
  p.unlink()
for folder in ['riesgo_total_s0','prefijo_nativo_s0','espacial_prior_cf_s0','espacial_fases_referencia_s0']:
 path=OUT/folder/'resultado.json';v=read(path)
 if any(t['file']==str((OUT/folder/'modelo.pt').relative_to(ROOT)) for t in removed):
  v['weights_retained']=False;v['cleanup_note']='Discarded full-cohort family: source, metrics and all predictions retained. No active/queued model depends on these weights.';write(path,v)
path=OUT/'piloto_maestro_probabilistico_s0/resultado.json';v=read(path);v['pilot_teacher_weights_retained']=False;v['cleanup_note']='Pilot teacher parameters and metrics retained; actual five full crossfit teachers are retained separately for the student.';write(path,v)
write(OUT/'limpieza_familia_generativa_terminada.json',dict(removed=removed,bytes_removed=sum(v['bytes'] for v in removed),all_predictions_and_metrics_retained=True,full_crossfit_teachers_and_parameters_retained=True,original_reference_unchanged=sha(BEST/'modelo.pt')=='55fa2262e3b26b259839fa06e6f28812e75ff30b578aceb6022c8286c2cb161f',Test='sealed'))
teacher=read(OUT/'maestro_cruzado_s0/resultado.json')
c=read(OUT/'CONTINUIDAD_OPERATIVA.json');c['teacher_CPU']=dict(experiment='maestro_cruzado_s0',complete=True,Train_n=1023485,whole_fold_excluded_for_every_row=True,parameters_sha256=teacher['parameters_sha256'],parameter_bytes=teacher['parameter_file_bytes'],scope='Privileged Train-only targets, not primary MAE')
c['completed_risk']=dict(validation=r['validation'],all_outputs=r['all_alpha_validation'],selected_alpha=r['selected_alpha'],steps=r['steps_run'],best_step=r['best_step'],new_candidates=[],weights_discarded=True)
c['compatibility_recovery']=dict(failed_before_updates='aux_solicitud_control_s0',replacement='aux_solicitud_control_s0_v2',standard_seeded_CUDA_restored=True,actual_GPU_gate_passed=True,reference_all127941_exact=True,archive='recuperacion_cumsum_v2')
c['auxiliary_queue']=dict(supervisor_pid=8560,session=32694,arms=['aux_solicitud_control_s0_v2','aux_solicitud_semantic_s0_v2'],sources_frozen=True)
c['soft_target_student_queue'].update(plan='plan_alumno_maestro.json',session=1486,waits_original_request_supervisor_pid=8560,run='alumno_maestro_s0_v2',sources_frozen=True)
c['next_after_result']=['Inspect actual current processes and pipeline_aux_solicitud_estado.json; risk and teacher CPU finished. Never restart retired PIDs.',
 'Request v2 control/semantic four-epoch study must pass/has passed actual initial GPU reference check. Its original strict-determinism attempt had zero optimizer updates and was archived.',
 'Student controller waits actual supervisor8560; teacher targets already complete and only Train. It then runs alumno_maestro_s0_v2 with literal-matched primary/sampling code and source/config checks.',
 'Read full point predictions, compare final fixed epoch against control and preserve any full candidate below131.06 plus any new best. Raw reproducibility before claims. Test sealed.',
 'If all fail, draw the actual scientific conclusion and continue a genuinely distinct justified alternative; do not repeat weights of the frozen failed generators. Goal confirmation73.5 remains unmet.']
write(OUT/'CONTINUIDAD_OPERATIVA.json',c)
marker='<!-- cierre_riesgo_y_recuperacion_GPU_v2 -->'
text="""

<!-- cierre_riesgo_y_recuperacion_GPU_v2 -->
El ensayo de riesgo de viaje completo terminó 3.174 actualizaciones (50.784 consultas), con checkpoint elegido en 3.000. Para pesos 0; 0,05; 0,2; 0,5; 1, los MAE completos fueron 131,256712; 131,308533; 131,769878; 134,244613; 142,366521 s. Ninguno mejora la referencia. El peso elegido en512 fue0,05: empeora en0,051821 s, IC95 % exploratorio por día [0,009255;0,095521]. No se amplía esta misma configuración solo para perseguir la cifra. No demuestra imposibilidad de todos los generativos.

El maestro cruzado completo terminó para1.023.485 filas, cada una predicha una sola vez sin ajustar su bloque semanal. Los parámetros comprimidos ocupan12.832.010 bytes. Su cobertura80 fue83,75 % sobre Train fuera de ajuste. Es supervisión privilegiada de Train, no un MAE OD. Se retiraron pesos generativos descartados y maestros piloto superados; se conservan todas sus predicciones, métricas y fuentes y los cinco maestros completos necesarios para el alumno.

El primer control auxiliar no actualizó pesos: falló al exigir un acumulado CUDA estrictamente determinista no implementado en esta instalación. Las fuentes y el fallo están archivados. La versiónv2 restablece el cálculo CUDA habitual con semillas fijas, sin cambiar pérdidas, etiquetas ni muestras. Su prueba GPU inicial pasó y reprodujo exactamente todas las predicciones de referencia antes de aprender. El control y la supervisión semántica mantienen cuatro épocas; el alumno de objetivos suaves espera su turno y usa exactamente el mismo control/objetivo primario/muestreo.

Estado vigente: pipeline_aux_solicitud_estado.json, pipeline_alumno_estado.json y CONTINUIDAD_OPERATIVA.json. Los identificadores antiguos de procesos no indican trabajos activos. Objetivo73,5 s pendiente; Test cerrado.
"""
for p in [OUT/'README.md',BASE/'RESUMEN_RESULTADOS.md']:
 body=p.read_text(encoding='utf-8-sig')
 if marker not in body:p.write_text(body+text,encoding='utf-8')
print(json.dumps(dict(removed_files=len(removed),removed_MiB=sum(v['bytes'] for v in removed)/1024**2,risk_qualified_candidates=0,goal='active',Test='sealed')))
