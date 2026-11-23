from base_replanteo import *
p=read(OUT/'politica_conservacion.json')
p['user_confirmation']='2026-11-22: preserve metrics and predictions also when MAE is greater than131.06 seconds.'
p['all_completed_experiments_including_MAE_above131_06']=True
p['discard_only']='Exact temporary caches/intermediates and discarded weights, preserving every completed metric, prediction archive, source, config, original best and every eligible qualifying model.'
write(OUT/'politica_conservacion.json',p)
c=read(OUT/'CONTINUIDAD_OPERATIVA.json')
c['current_trial']=dict(experiment='subviajes_train_s0',stage='Train-only bounded auxiliary-data preparation; GPU idle',process_status='See actual process/log; no stale field process')
c['completed_field']=dict(id='campo_multiescala_s0',epochs=20,best_mae_s=135.42382817079746,verdict='Negative; conditional capacity control skipped; all epoch metrics/predictions preserved')
c['parent_initialization']=dict(id=BEST.name,sha256=sha(BEST/'modelo.pt'),protected=True)
c['resource_policy']='One heavy GPU job; bounded <=500k auxiliary examples, compressed216feature cache <=600MiB, no query route/count input'
c['next_after_result']=['Verify Train-only subtrip data provenance and whole-parent week exclusion.','Separate auxiliary duration head sharing original encoder; original primary labels and query unchanged.','Freeze exact sources before GPU; reproduce reference and finite gradients; execute predeclared training.','Keep every completed metric/prediction even if worse; best weights and all complete MAE<131.06 candidates preserved.','Goal still unmet; Test remains sealed.']
write(OUT/'CONTINUIDAD_OPERATIVA.json',c)
print('Conservation and active-trial state updated; all completed metrics/predictions retained.')
