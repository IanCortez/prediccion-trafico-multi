"""Independent result inventory and critical conclusion, never declares the goal achieved."""
from base_replanteo import *
access=deny_test();run=OUT/'progreso_vial_recurrente_s0';bf=run/'etapa_B'
a=read(run/'resultado_etapa_A.json');b=read(bf/'resultado.json');sm=read(run/'resumenes_publicos/resultado.json')
assert a['complete'] and b['complete'] and sm['complete']
z=dict(np.load(bf/'validacion.npz'));d=dict(np.load(DATA/'validation_x.npz'));truth=np.load(DATA/'validation_y.npy')
assert len(truth)==127941 and np.array_equal(z['true'],truth)
for k in ['row','timestamp']:assert np.array_equal(d[k],z[k])
assert sha(bf/'validacion.npz')==b['predictions_sha256']
for name,m in b['metrics'].items():assert metric(truth,z[name])==m
r=np.load(run/'validacion_solo_raiz.npz');assert metric(truth,r['pred'])==a['root_diagnostic_validation']
for name,entry in b['models'].items():
 if entry['preserved']:assert sha(bf/entry['file'])==entry['sha256']
old=read(OUT/'nuevo_mejor_verificado.json')
numerical=min(b['metrics'],key=lambda k:b['metrics'][k]['mae_s'])
improved=b['metrics'][numerical]['mae_s']<old['validation']['mae_s']
certificate=run/('certificacion_'+numerical)/'resultado.json'
cert=read(certificate) if certificate.exists() else None
if improved:assert cert and cert['passed'],'Newnumericalminimumrequiresindependentreplaybeforeconclusion'
decision=('Useful predeclared fullpipelinegate passed; next matchedprefix-resetcontrol andindependentconfirmation eligibility, noTestopeningnow.'
 if b['meaningful_gate_passed'] else
 'Fixed fullsequence experiment failed meaningfulgate; no epoch/stylecount/stepcap retuning. Examine trainqueryexposuregap/coverage before choosing a different hypothesis.')
conclusion=dict(complete=True,scope='FullunchangedprimaryODValidation, reusedduringadaptive research; not independentconfirmation.',
 primary_rows_and_times_unchanged=True,root_diagnostic=a['root_diagnostic_validation'],
 full_pipeline_metrics=b['metrics'],comparisons=b['comparisons'],meaningful_gate_passed=b['meaningful_gate_passed'],
 numerical_minimum_model=numerical,numerical_minimum=b['metrics'][numerical],
 candidate_better_than_protected=bool(improved),candidate_certification=cert,decision=decision,
 route_diagnostics={k:{f:v for f,v in info.items() if f in [
  'n','omitted_pair_mass_mean','weighted_tail_mass_mean','weighted_completion_mass_mean','weighted_steps_mean','style_eta_dispersion_s_mean']}
  for k,info in sm['outputs'].items()},
 limits=['Train road/time labels inferred between 15-second GPS samples; not exact annotated physical paths or idling.',
 'Unsupported interior chain turns reset private prefix supervision; all original primary trips/totalT kept.',
 'Fixed128 teacher fragments truncate long recurrent history; querycap192 independentofactual N.',
 'Training uses observedteacher clock/prefix; queryuses its own priorstyles/generatedprefix/predictedclock. Exposuregap remainsa risk.',
 'One seed, reusedValidation.73.5confirmation not reached orclaimed.'],
 preservation='Allsixcompletedprimaryoutputs(5trees+root), originalrows/labels/quantiles/config/sources retained, including >131.06. Qualifiedweights anddependencies retained.',
 protected_best=old['validation'],goal='ACTIVE',Test='sealed')
write(OUT/'conclusion_progreso_recurrente.json',conclusion)
state=read(OUT/'CONTINUIDAD_OPERATIVA.json')
state['current_trial']=dict(experiment=run.name,stage='Fullprimaryexperimentcomplete; inspectcriticalconclusionandcontinuewithnextdistincthypothesis.',pid=None,Test='sealed')
state['latest_recurrent_experiment']=conclusion
state['next_after_result']=[decision,'Do not repeat failedwidth/stylecount/bin/epoch grids.',
 'If any numericalcandidate beatsprotectedbest, it is certified but registry promotion requires honestlabeling; root/q50/controlposthoc improvement not semanticcausality.',
 'Preserveallcompletedoutputs includingabove131.06. Best/40qualifiedhistorymodels/TransferTrajprotected; independentconfirmation73.5notyetmet.']
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
summary=BASE/'RESUMEN_RESULTADOS.md';text=summary.read_text(encoding='utf-8-sig')
marker='### Secuencias completas y reloj previsto (Porto OD)'
assert marker not in text
lines=['',marker,'','Experimento de seis épocas con todos los viajes originales de Train; secuencias auxiliares corregidas, memoria recurrente y reloj previsto. Test cerrado.','',
 '| Modelo | MAE Validation (s) |','|---|---:|',
 '| Raíz sin recorrido previsto (diagnóstico) | '+format(a['root_diagnostic_validation']['mae_s'],'.6f')+' |']
for name,m in b['metrics'].items():lines.append('| '+name+' | '+format(m['mae_s'],'.6f')+' |')
lines+=['','Resultado del criterio de mejora: '+str(b['meaningful_gate_passed'])+'.',
 'Se conservan todas las métricas y predicciones, incluso las que superan 131,06 s. Las calles y tiempos auxiliares son inferidos; este resultado no constituye una confirmación independiente ni una reproducción exacta de UNITE.',
 'La meta de confirmación inferior a 73,5 s continúa pendiente.']
summary.write_text(text+'\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(complete=True,metrics=b['metrics'],meaningful_gate=b['meaningful_gate_passed'],goal='ACTIVE',Test='sealed')),flush=True)
