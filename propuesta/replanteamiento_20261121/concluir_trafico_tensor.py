"""Conclude failed field gate and record the distinct distributional alternative."""
from base_replanteo import *
access=deny_test();run=OUT/'piloto_trafico_tensor_train_s0';r=read(run/'resultado.json');assert not r['passed']
z=dict(np.load(run/'predicciones_held_Train.npz'));assert sha(run/'predicciones_held_Train.npz')==r['predictions_sha256']
assert metric(z['true'],z['pred'])==r['metrics'] and metric(z['true'],z['control'])==r['control']
weights=read(run/'pesos_descartados.json');assert len(weights['removed'])==6 and all(not (run/q['file']).exists() for q in weights['removed'])
conclusion=dict(complete=True,scope=r['scope'],metrics=r['metrics'],control=r['control'],comparison=r['comparison'],
 decision='Failed>=1secondgate; no matchedparentclockcontrol and no rank/slot/offsettuning. Keepallmetrics/predictions/sources; sixnegativefieldweights alreadydiscarded.',
 limits='Endpointpooledfieldfeatures did not improve thiscontrol. Doesnot disproveeverytime-dependentroadmodel orforecast; causalclockalignmentgain notestablished.',
 six_source_views_parentfold_excluded=r['all_own_parentfold_parameters_and_support_excluded'],
 next='IndependentmonotoneSplineCDF family. Usesonly206publicraw5features andunchangedoriginaldurationbins, no road/driver auxiliaries.',
 Test='sealed')
write(OUT/'conclusion_trafico_tensor.json',conclusion)
removed=[]
for p,expected in [(run/'caracteristicas_Train.npz',r['features_sha256']),
 (OUT/'datos_trafico_tensor_train_s0/observaciones_Train.npz',read(OUT/'datos_trafico_tensor_train_s0/resultado.json')['observations_sha256'])]:
 assert p.resolve().is_relative_to(OUT.resolve()) and sha(p)==expected
 removed.append(dict(file=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=expected,kind='Regenerablederivedfeatures/Trainauxiliaryobservations, notpredictions.'))
write(OUT/'limpieza_trafico_tensor.json',dict(removed=removed,all_completed_metrics_predictions_preserved=True))
for entry in removed:(ROOT/entry['file']).unlink()
s=read(OUT/'CONTINUIDAD_OPERATIVA.json');s['current_trial']=dict(experiment='piloto_spline_cdf_train_s0',stage='Fixed50epochTrain200k/held50k, heldonlyfinal. Newconditionalmonotonespline distribution; CPUinverse/monotonicity/probabilitytests passed.',pid=None,log='piloto_spline_cdf_train.log',Test='sealed')
s['completed_new_instruments']['temporal_tensor_gate']=conclusion;s['primary_audit']=dict(outputs=240,archives=28,reported_metric_max_error=0.,Test=False)
s['next_after_result']=['FollowSplineCDF50fixedepochs; do not editactivefrozenmodel/runner and do notlaunch anotherheavyGPUjob.',
 'Atleast1secondheldTrain gainandupperCI<0required before fullprimarySplineCDFtrial. No sourcequeryinformation orlabelchanges.',
 'Do not repeatfailedamortizedstreetvalue, generatedroute,driveroracle orquarterhourtrafficfields; preserveallpointmetrics/predictions.',
 'Currentbest130.158 and40qualifyingweightsstillprotected;73.5goalactive andTestsealed.']
write(OUT/'CONTINUIDAD_OPERATIVA.json',s)
report=BASE/'RESUMEN_RESULTADOS.md';text=report.read_text(encoding='utf-8-sig')
old='Sigue una prueba de rutas efectivamente generadas desde origen, destino y hora, sin rutas reales en la reserva.'
new='La prueba OD con rutas generadas no mejoró el control interno (133,969 frente a 133,355 s); el campo de tráfico por hora tampoco lo mejoró (135,506 frente a 135,464 s). Estos son ensayos dentro de Train y no cifras de validación primaria. Continúa una familia distinta de distribuciones temporales mediante splines monótonos, con las mismas entradas y etiquetas.'
assert text.count(old)==1;report.write_text(text.replace(old,new),encoding='utf-8')
print(json.dumps(dict(concluded=True,removed_bytes=sum(q['bytes'] for q in removed),next='SplineCDFTrainpilot',all_completed_predictions_preserved=True)),flush=True)
