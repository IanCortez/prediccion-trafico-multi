from base_replanteo import *
run=OUT/'historica_fresca_s0';history=read(run/'historia.json');rows=[]
for row in history:
 if not row['preserved']:continue
 path=run/('epoca_'+str(row['epoch'])+'.pt');assert path.resolve().is_relative_to(OUT.resolve()) and path.exists()
 rows.append(dict(epoch=row['epoch'],validation=row['validation'],model_file=str(path),model_sha256=sha(path),qualifies_below131_06=row['validation']['mae_s']<131.06))
best=min(history,key=lambda z:z['validation']['mae_s'])
record=dict(experiment=run.name,scope='Complete unchangedValidation observed during active training, not confirmation or frozen-candidate reproduction.',
 candidates=rows,best_observed_epoch=best['epoch'],best_observed_validation=best['validation'],
 latest_completed_epoch=history[-1]['epoch'],all_epoch_metrics_predictions_retained=True,
 final_goal73_5_achieved=False,Test_access=False)
write(OUT/'candidatos_vivos_historica.json',record)
state=read(OUT/'CONTINUIDAD_OPERATIVA.json')
state['current_trial'].update(latest_observed_completed_epoch=history[-1]['epoch'],best_observed_epoch=best['epoch'],best_observed_validation=best['validation'])
state['provisional_candidate']=dict(id=run.name,epoch=best['epoch'],validation=best['validation'],
 status='Checkpoint preserved, below131.06; active experiment still running. Do not promote registry before controlled completion/certification.',
 manifest='candidatos_vivos_historica.json')
state['mechanism_probe']=read(run/'diagnostico_uso_historia_epoca_15.json')
state['previous_goal_turn_classification']='PROGRESS and verified waits: established/certified131.125167reference, built leakage-checked historical prototypes, launched fresh model, observed/preserved sub131.06candidates.'
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
p=OUT/'verificar_conservacion.py';code=p.read_text(encoding='utf-8-sig')
needle="write(OUT/'inventario_conservacion.json',report);print(json.dumps({k:report[k] for k in ['project_GiB','file_count','new_work_MiB','best_model_unchanged','current_best_unchanged','current_best_mae_s','seconds']}))"
replacement="""livefile=OUT/'candidatos_vivos_historica.json'
if livefile.exists():
 live=read(livefile);report['live_qualified_candidates']=[]
 for candidate in live['candidates']:
  path=Path(candidate['model_file']).resolve()
  assert path.is_relative_to(OUT.resolve()) and sha(path)==candidate['model_sha256']
  report['live_qualified_candidates'].append(dict(epoch=candidate['epoch'],mae_s=candidate['validation']['mae_s'],sha256=candidate['model_sha256'],unchanged=True))
write(OUT/'inventario_conservacion.json',report);print(json.dumps({k:report[k] for k in ['project_GiB','file_count','new_work_MiB','best_model_unchanged','current_best_unchanged','current_best_mae_s','seconds']}))"""
if needle in code:p.write_text(code.replace(needle,replacement),encoding='utf-8')
else:assert "livefile=OUT/'candidatos_vivos_historica.json'" in code
print(json.dumps(dict(latest_epoch=history[-1]['epoch'],best_observed=best['validation'],preserved_models=len(rows),qualifying_models=sum(r['qualifies_below131_06'] for r in rows))))
