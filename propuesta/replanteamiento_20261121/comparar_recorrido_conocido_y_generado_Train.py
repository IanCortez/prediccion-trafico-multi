"""Train-only matched scenario diagnostic after full Train public bank; never a primary score."""
from base_replanteo import *
access=deny_test();start=time.time();run=OUT/'progreso_vial_recurrente_s0';sf=run/'resumenes_publicos'
info=read(sf/'train_resultado.json');p=sf/'train_resumenes.npz';assert sha(p)==info['sha256']
z=dict(np.load(p));d=dict(np.load(DATA/'train_x.npz'));y=np.load(DATA/'train_y.npy')
for k in ['row','timestamp']:assert np.array_equal(z[k],d[k])
s=z['summary'];eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index']
full={}
for name,col in [('root',1),('public_mixture_median',14),('public_weighted_mean',3)]:
 pred=s[eligible,col].astype(np.float64).clip(60,1785)
 full[name]=dict(**metric(y[eligible],pred),mean_signed_error_s=float(np.mean(pred-y[eligible])))
private=dict(np.load(OUT/'diagnostico_suma_tiempos_progreso_Train.npz'));ids=private['train_index']
assert np.array_equal(private['original_total_T'],y[ids])
ratio=private['known_time_true']/y[ids]
complete=np.abs(ratio-1)<.00001
matched={}
for label,mask in [('all512',np.ones(len(ids),bool)),('known_time_complete_diagnostic_subgroup',complete)]:
 t=y[ids[mask]];out={}
 for name,pred in [('root',s[ids[mask],1]),('public_mixture_median',s[ids[mask],14]),
  ('public_weighted_mean',s[ids[mask],3]),('private_teacher_prefix_clock_style',private['known_time_pred'][mask])]:
  pred=np.asarray(pred,np.float64).clip(60,1785)
  out[name]=dict(**metric(t,pred),mean_signed_error_s=float(np.mean(pred-t)))
 matched[label]=out
result=dict(complete=True,scope='TRAIN diagnostics only; not primaryValidation orconfirmation; no newcheckpoint qualifiesfromthesevalues.',
 all_primary_Train_queries=1023485,public_forecasts_Train=full,matched512_scenarios=matched,
 complete_time_subgroup_n=int(complete.sum()),
 scenario_difference='Private usesGTroad/prefix/clock/recognizedstyle; public usesgeneratedroute/clock, priorstylemixture andattachmentfallback. Differencecannotbeattributedtoroutealone.',
 known_time_subgroup_warning='Diagnostic-onlygroup withinfixed512, never changesprimarycohort/labels ormodelselection threshold.',
 sources_sha256=sha(__file__),public_bank_sha256=info['sha256'],
 private_predictions_sha256=sha(OUT/'diagnostico_suma_tiempos_progreso_Train.npz'),
 no_Validationlabels_or_Test=True,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(OUT/'comparacion_recorrido_conocido_y_generado_Train.json',result);print(json.dumps(result),flush=True)
