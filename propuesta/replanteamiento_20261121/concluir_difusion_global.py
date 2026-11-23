"""Independent all-row primary audit for global joint diffusion, never claims goalcompletion."""
from base_replanteo import *
access=deny_test();run=OUT/'difusion_global_tiempo_geometria_s0';r=read(run/'resultado.json');assert r['complete']
z=dict(np.load(run/'validacion.npz'));d=dict(np.load(DATA/'validation_x.npz'));y=np.load(DATA/'validation_y.npy')
assert len(y)==127941 and np.array_equal(z['true'],y) and sha(run/'validacion.npz')==r['predictions_sha256']
for k in ['row','timestamp']:assert np.array_equal(z[k],d[k])
assert metric(y,z['pred'])==r['validation'] and np.array_equal(z['quantiles'][:,1],z['pred'])
sensitivity=None;certificate=None
if r['qualified']:
 assert sha(r['model_file'])==r['model_sha256']
 certificate=read(run/'certificacion_raw_y_cohorte.json');assert certificate['passed']
 f=run/'sensibilidad_256.npz';a=dict(np.load(f));sensitivity=read(run/'sensibilidad_256.json')
 for k in ['row','timestamp']:assert np.array_equal(a[k],d[k])
 assert np.array_equal(a['true'],y) and metric(y,a['pred'])==sensitivity['validation']
uncertainty=dict(coverage80=float(np.mean((y>=z['quantiles'][:,0])&(y<=z['quantiles'][:,2]))),
 mean_width80_s=float(np.mean(z['quantiles'][:,2]-z['quantiles'][:,0])),
 interpretation='EmpiricalonreusedValidation; not a guaranteedcoverageclaim.')
decision=('Usefulfixed64samplegain: review256sensitivity andrunmatchedrandomgeometrycontrol beforesemanticcontributionclaims.'
 if r['meaningful_gate_passed'] else
 'Globaljointdiffusiondidnotpassusefulnessgate. Do notretuneparticlecounts/epochs/widths againstsameValidation; usecriticalerroranalysis foradifferentmechanism.')
result=dict(complete=True,validation=r['validation'],comparison=r['paired_vs_reference'],meaningful_gate_passed=r['meaningful_gate_passed'],
 qualified=r['qualified'],sensitivity256=sensitivity,certificate=certificate,uncertainty=uncertainty,decision=decision,
 limitations=['Geometriclabels areobservedTrainroadfingerprints, notexactannotatedpaths.',
 'Unknowngeometryusesa normalizedplaceholderandmaskedgeometryscore; timevelocitylosskeepsalloriginalTrainqueries. Thisdoesnotestablishanexactmissing-datajointlikelihood.',
 'Jointglobaldenoising avoidsautoregressiveroadprefixqueries butdenoisingerror/finitedraws canstilllimitaccuracy.',
 'SingleNNseed/reusedValidation. Independentconfirmation73.5remainsunproven.'],
 all_primary_rows_labels_andquery_information_unchanged=True,all_completed_metrics_predictions_preserved=True,goal='ACTIVE',Test='sealed')
write(OUT/'conclusion_difusion_global.json',result)
state=read(OUT/'CONTINUIDAD_OPERATIVA.json');state['current_trial']=dict(experiment=run.name,stage='Completeprimaryscoreandcriticalconclusion; continuegoal.',pid=None,Test='sealed')
state['latest_global_diffusion']=result;state['next_after_result']=[decision,
 'Ifanynumericalminimum exists, it remainsapendingcandidateuntil source/checkpoint/rawreplay/sensitivity reviewandbestregistry/recipe update.',
 'NeveropenTestuntilaneligiblecandidateisfrozen. Keepalloriginals/qualifiedmodels/TransferTraj andallcompletedoutputs includingabove131.06.']
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
p=BASE/'RESUMEN_RESULTADOS.md';s=p.read_text(encoding='utf-8-sig');title='### Difusión conjunta global de tiempo y geometría'
assert title not in s
s+='\n'+title+'\n\nModelo nuevo entrenado durante 30 épocas fijas con todos los tiempos originales; la geometría observada de Train es supervisión auxiliar. La consulta genera conjuntamente tiempo y geometría desde ruido, sin recibir recorrido real ni cantidad de puntos.\n\n'
s+='MAE Validation con 64 muestras y 32 pasos de denoising: '+format(r['validation']['mae_s'],'.6f')+' s. Criterio de mejora útil: '+str(r['meaningful_gate_passed'])+'.\n\n'
s+='Se conservaron todas las predicciones y métricas originales. Los candidatos elegibles incluyen reproducción desde cinco campos y sensibilidad a 256 muestras. Objetivo de confirmación 73,5 s pendiente; Test cerrado.\n'
p.write_text(s,encoding='utf-8')
print(json.dumps(dict(complete=True,validation=r['validation'],gate=r['meaningful_gate_passed'],qualified=r['qualified'],goal='ACTIVE',Test='sealed')),flush=True)
