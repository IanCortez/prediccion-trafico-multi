"""Close/replay the smallpilot; retire only its verified negative transient weights."""
from base_replanteo import *
run=OUT/'mezcla_individual_mediana_Train_s0';r=read(run/'resultado.json');replay=read(run/'reproduccion_independiente.json')
assert r['complete'] and replay['passed'] and not replay['stable_practical_gate_passed']
assert read(OUT/'preflight_balanceado_GPU.json')['passed']
assert sha(run/'predicciones_held_Train.npz')==r['predictions_sha256']
assert sha(run/'predicciones_balanceadas_held_Train.npz')==replay['stable_predictions_sha256']
for name,digest in r['sources'].items():assert sha(run/'fuentes'/name)==digest
gaps={}
for mode,metric_values in replay['stable_metrics'].items():
 h=read(run/mode/'historia.json');assert len(h)==80 and all(v['train_n']==40000 for v in h)
 gaps[mode]=metric_values['mae_s']-h[-1]['train_mae_s']
 with np.load(run/mode/'predicciones_held_Train.npz') as z:assert metric(z['true'],z['pred'])==r['metrics'][mode]
 a=read(run/mode/'resultado.json');a['held_prediction_sha256']=sha(run/mode/'predicciones_held_Train.npz');write(run/mode/'resultado.json',a)
assert gaps['none']>20
retirement_file=run/'retencion_pesos_descartados.json';previous=read(retirement_file) if retirement_file.exists() else None
items=[]
for item in r['checkpoints']:
 path=Path(item['file']).resolve();assert path.is_relative_to(run.resolve())
 if path.exists():
  assert sha(path)==item['sha256'];items.append(dict(file=str(path),sha256=item['sha256'],bytes=path.stat().st_size,removed=False))
 else:
  assert previous and any(v['file']==str(path) and v['sha256']==item['sha256'] and v['removed'] for v in previous['weights'])
  items.append(next(v for v in previous['weights'] if v['file']==str(path)))
write(retirement_file,dict(reason='All3finalsmallpilotmodels failpracticalgate, are above131.06 evenonheldTrain, and independentfull/rawreplay has passed; metrics/predictions/config/sources protected.',weights=items))
for item in items:
 if not item['removed']:
  path=Path(item['file']);assert path.resolve().is_relative_to(run.resolve()) and sha(path)==item['sha256'];path.unlink();item['removed']=True
write(retirement_file,dict(complete=True,weights=items,bytes_released=sum(v['bytes'] for v in items),all_completed_predictions_preserved=True,only_negative_transient_pilot_weights_removed=True))
conclusion=dict(complete=True,stable_metrics=replay['stable_metrics'],stable_comparisons=replay['stable_comparisons'],practical_gate=False,
 generalization_gap_s=gaps,independent_full10000_and_raw65_replay_passed=True,original_fullquantiles_reproduced_exactly=True,
 precision_changed_MAE_max_s=max(abs(replay['stable_metrics'][m]['mae_s']-r['metrics'][m]['mae_s']) for m in r['metrics']),
 interpretation='Historical time helps versus matchedneural/geometry controls, but all3models have verylarge train/heldgaps and lose to the storedtree reference. This is consistent with overfitting/temporalshift, not evidence of a universal informationbound.',
 next='Activate the predeclared fullprecutoff951718Trainrow4epoch comparison after sourceconstruction checks, using stablemedian/NLL. Sameheld10k and same gates; no targetcohort/querychange.',
 original_and_stable_metrics_predictions_configs_sources_retained=True,negative_weights_retired_after_replay=True,
 goal='ACTIVE',Test='SEALED',Validation=False)
write(run/'conclusion.json',conclusion)
r['independent_replay_pending']=False;r['independent_replay_sha256']=sha(run/'reproduccion_independiente.json')
r['retention_record']=retirement_file.name;write(run/'resultado.json',r)
state=read(OUT/'CONTINUIDAD_OPERATIVA.json');state['current_trial']=None
state['last_closed_timepilot_id']=run.name;state['last_closed_timepilot']=conclusion
state['next_after_result']=[
 'OriginalNN40k80epoch experiment terminal, independentlyreplayed, allmetrics/preds preserved; only3negativeweights retired by exactSHA.',
 'Activate preparar_indices_escala_Train.py:951718fit+10000held, same originalsourcecutoff andbefore-treeparentweekexclusion. NoGPU needed for construction; cachecap350MiB, no rawtokens.',
 'Then implement a streamed all-data4epoch driver withstable median/binNLL; publicX and compactsourceindices onGPU, originalpointbank/prefix reused; no two all-data rawtoken tensors.',
 'GPUstable-backward preflight passed withfit256only, no optimizerstep or heldlabeltraining. CPUadversarial signedlogtail/gradient checks passed.',
 'Keep >=5s matchedbaseline/reference and>=2s geometrycontrol gate unchanged. No primaryValidation until a positive Train gate. Bestprimary129.990034,73.5notreached,Testsealed.'
]
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
active=read(OUT/'arquitectura_activa.json');active.update(current_experiment=None,current_protocol=None,last_closed_timepilot=conclusion)
write(OUT/'arquitectura_activa.json',active)
marker='<!-- resultado_mezcla_individual_Train_s0 -->'
for name in ['RESUMEN_RESULTADOS.md','ARQUITECTURA.md']:
 p=BASE/name;s=p.read_text(encoding='utf-8')
 if marker not in s:
  s+='\n\n'+marker+'\n## Mezcla individual: piloto de 40.000 viajes cerrado\n\n'
  s+='Tras 80 épocas fijas, los MAE de la reserva Train fueron 150,527837 s sin fuentes, 148,302471 s con el control geométrico y 145,292730 s con tiempos observados. La fuente temporal mejoró 5,24 s frente a la red sin fuentes y 3,01 s frente al control geométrico, pero empeoró 11,94 s frente a la referencia de árboles de 133,355393 s. No cumple el criterio de expansión.\n\n'
  s+='La reproducción independiente coincidió en todos los cuantiles de los 10.000 viajes y en 65 consultas crudas. La mediana estable alteró el MAE menos de 0,000004 s; el problema numérico extremo no explica este fallo predictivo. Las grandes diferencias frente al error de ajuste son compatibles con sobreajuste y cambio temporal. Se conservaron las métricas y predicciones de ambas evaluaciones, las configuraciones, fuentes e historias; se retiraron solo los tres pesos transitorios descartados después de reproducirlos.\n\n'
  s+='Se activa la comparación previamente definida con 951.718 viajes anteriores al mismo corte y cuatro épocas completas, aproximadamente el mismo presupuesto de actualizaciones. Mantiene los mismos 10.000 viajes de reserva, filtros y tiempos originales, y no modifica los campos de consulta. Test y la validación primaria siguen cerrados para esta familia.\n'
  p.write_text(s,encoding='utf-8')
print(json.dumps(dict(closed=run.name,stable_metrics=replay['stable_metrics'],generalization_gap_s=gaps,bytes_released=sum(v['bytes'] for v in items),all_predictions_preserved=True)))
