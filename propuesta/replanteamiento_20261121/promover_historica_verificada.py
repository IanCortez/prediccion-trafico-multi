"""Promote only a fully reproduced frozen checkpoint; control reference stays archived."""
from base_replanteo import *
run=OUT/'historica_fresca_s0';r=read(run/'resultado.json');cert=read(run/'certificacion/resultado.json')
assert cert['passed'] and cert['full127941saved_point_predictions_exact'] and cert['raw65_public_and_history_features_exact']
assert sha(r['model_file'])==r['model_sha256']==cert['checkpoint_sha256']
old=read(OUT/'nuevo_mejor_verificado.json');frozen_ref=read(run/'referencia_previa.json')
assert frozen_ref['model_sha256']==r['config']['reference_checkpoint_sha256']
assert r['validation']['mae_s']<old['validation']['mae_s']
lineagefile=OUT/'modelos_mejores_protegidos.json'
lineage=read(lineagefile) if lineagefile.exists() else []
for item in [old]:
 if not any(x['model_sha256']==item['model_sha256'] for x in lineage):lineage.append(item)
best=dict(id=run.name,epoch=r['best_epoch'],validation=r['validation'],model_file=r['model_file'],model_sha256=r['model_sha256'],
 certification=str((run/'certificacion/resultado.json').relative_to(ROOT)),
 scope='Fully reproduced127941Validation and raw65fivefield queries; one seed and reusedValidation, not confirmation.',
 qualifies_below131_06=True,preserve_original_reference=old['preserve_original_reference'],
 historical_memory_sha256=read(OUT/'datos_prototipos_historicos_s0/resultado.json')['memory']['sha256'],
 all_completed_metrics_predictions_retained=True,control='Running with frozen pretrial reference; mechanism conclusion pending.')
if not any(x['model_sha256']==best['model_sha256'] for x in lineage):lineage.append(best)
for item in lineage:
 p=Path(item['model_file']).resolve();assert p.is_relative_to(OUT.resolve()) and sha(p)==item['model_sha256']
write(lineagefile,lineage);write(OUT/'nuevo_mejor_verificado.json',best)
rows=[]
for item in r['saved_checkpoints']:
 p=run/item['file'];assert sha(p)==item['sha256']
 rows.append(dict(epoch=item['epoch'],validation=item['validation'],model_file=str(p),model_sha256=item['sha256'],qualifies_below131_06=item['threshold_qualifies']))
write(OUT/'candidatos_vivos_historica.json',dict(experiment=run.name,stage='Treatment complete;20threshold candidates preserved; best checkpoint reproduced.',
 candidates=rows,best_observed_epoch=r['best_epoch'],best_observed_validation=r['validation'],latest_completed_epoch=r['epochs_run'],
 all_epoch_metrics_predictions_retained=True,final_goal73_5_achieved=False,Test_access=False))
state=read(OUT/'CONTINUIDAD_OPERATIVA.json');state['goal_status']='ACTIVE; verifiedValidation130.204821s; final73.5confirmation unmet; Testsealed'
state['retained_best']=best;state['completed_historical_treatment']=dict(epochs=r['epochs_run'],validation=r['validation'],best_epoch=r['best_epoch'],full_raw_certification=True,saved_threshold_models=len(rows))
state['current_trial']=dict(experiment='control_historica_fresca_s0',pid=24952,exec_session_id=10237,
 stage='One GPU; identical initial model hash and algorithm, historical branch unavailable; separate bounded-retention control runner',sources_frozen=True,Test='sealed')
state['next_after_result']=['Inspect actual control24952/session10237 and control_historica_fresca_s0.log; no blind restart after timeout.','Control uses entrenar_historica_fresca_control.py and frozen historical main/referencia_previa.json. Registry promotion cannot change its reference.','Do not edit active/queued sources. Let predeclared min20,max50,3600secondcap finish.','Certify its retained best weights, compare fixed epoch20 and selected models on identical rows; conditional ablation is not equal effective capacity.','All current20sub131.06checkpoints and verified best130.204821s are protected by hashes. Keep original model and TransferTraj.','No goal completion:73.5confirmation remains unmet; continue a justified distinct improvement after conclusions.']
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
start='<!-- estado_actual_verificado_inicio -->';end='<!-- estado_actual_verificado_fin -->'
header=f"""{start}
Resultado vigente verificado: **MAE130,204821s**, arquitectura histórica entrenada desde cero, época27. Terminó39épocas. Se reprodujeron exactamente127941predicciones y las consultas crudas65, individuales y en distinto orden. Se conservan20candidatos menores de131,06s. El control sin rama histórica está en ejecución; todavía no se atribuye toda la diferencia a esa rama.

La meta73,5s sigue pendiente y Test cerrado. El resultado es exploratorio: una semilla, validación reutilizada y una cohorte diferente deTransferTraj. Ver nuevo_mejor_verificado.json, historica_fresca_s0/certificacion/resultado.json y control_historica_fresca_s0.log.
{end}"""
for dest in [OUT/'README.md',BASE/'RESUMEN_RESULTADOS.md']:
 text=dest.read_text(encoding='utf-8-sig');a=text.index(start);b=text.index(end)+len(end);dest.write_text(text[:a]+header+text[b:],encoding='utf-8')
p=OUT/'verificar_conservacion.py';code=p.read_text(encoding='utf-8-sig')
needle="livefile=OUT/'candidatos_vivos_historica.json'"
addition="""lineagefile=OUT/'modelos_mejores_protegidos.json'
if lineagefile.exists():
 report['verified_best_lineage_unchanged']=[]
 for model in read(lineagefile):
  path=Path(model['model_file']).resolve()
  assert path.is_relative_to(OUT.resolve()) and sha(path)==model['model_sha256']
  report['verified_best_lineage_unchanged'].append(dict(id=model['id'],sha256=model['model_sha256'],unchanged=True))
"""
assert needle in code;code=code.replace(needle,addition+needle);p.write_text(code,encoding='utf-8')
print(json.dumps(dict(verified_best=best['validation'],saved_threshold_models=len(rows),control_reference_unchanged=True,Test_access=False)))
