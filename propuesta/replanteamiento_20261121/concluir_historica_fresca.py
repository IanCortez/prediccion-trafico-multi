"""Compare certified historical treatment and its same-initialization branch ablation."""
from base_replanteo import *
start=time.time();access=deny_test()
ids=['historica_fresca_s0','control_historica_fresca_s0']
results={};certs={};arrays={}
for name in ids:
 run=OUT/name;r=read(run/'resultado.json');c=read(run/'certificacion/resultado.json')
 assert c['passed'] and c['checkpoint_sha256']==r['model_sha256']
 assert c['full127941saved_point_predictions_exact'] and c['raw65_medians_match_full_saved'] and c['reverse_and_singleton_all_quantiles_exact']
 assert sha(r['model_file'])==r['model_sha256'] and r['epochs_run']>=20
 z=dict(np.load(run/'validacion.npz'))
 assert metric(z['true'],z['pred'])==r['validation']
 results[name]=r;certs[name]=c;arrays[name]=z
a,b=[arrays[n] for n in ids];ra,rb=[results[n] for n in ids]
for key in ['row','timestamp','true']:assert np.array_equal(a[key],b[key])
assert ra['initial_model_state_sha256']==rb['initial_model_state_sha256']
for key in ['seed','batch','lr','weight_decay','clip','geometry_weight','rps_weight','canonical_inference_batch','inference_candidate_chunk','initial_historical_gate_logit','base_constructor_sha256','history_constructor_sha256','map_normalization_sha256']:
 assert ra['config'][key]==rb['config'][key],key
fixed={name:metric(arrays[name]['true'],arrays[name]['epoch_20']) for name in ids}
fixed_comparison=paired(a['true'],a['epoch_20'],b['epoch_20'],a['timestamp'])
selected_comparison=paired(a['true'],a['pred'],b['pred'],a['timestamp'])
reference=read(OUT/'historica_fresca_s0/referencia_previa.json')
previous=dict(np.load(OUT/reference['id']/'validacion.npz'))
for key in ['row','timestamp','true']:assert np.array_equal(a[key],previous[key])
main_vs_previous=paired(a['true'],a['pred'],previous['pred'],a['timestamp'])
protected=[]
for name,r in results.items():
 for item in r['saved_checkpoints']:
  path=(OUT/name/item['file']).resolve();assert path.is_relative_to((OUT/name).resolve())
  qualifies=item['validation']['mae_s']<131.06
  if qualifies or path.is_file():
   assert path.is_file() and sha(path)==item['sha256']
   if qualifies:protected.append(dict(experiment=name,epoch=item['epoch'],file=str(path),sha256=item['sha256'],validation=item['validation']))
write(OUT/'modelos_historicos_completos_protegidos.json',dict(candidates=protected,preserve=True,all_completed_metrics_predictions_retained=True,Test='sealed'))
verdict='Historical branch improves the matched fixed20 epoch comparison.' if fixed_comparison['ci95_s'][1]<0 else 'Historical branch advantage is not established by the matched fixed20 epoch comparison.'
result=dict(treatment=ra['validation'],control=rb['validation'],fixed_epoch20=fixed,
 fixed_epoch20_paired_treatment_minus_control=fixed_comparison,
 selected_epoch_paired_treatment_minus_control=selected_comparison,
 selected_treatment_minus_previous_verified=main_vs_previous,
 initial_state_sha256=ra['initial_model_state_sha256'],same_initial_weights=True,
 treatment_epochs=ra['epochs_run'],control_epochs=rb['epochs_run'],
 treatment_best_epoch=ra['best_epoch'],control_best_epoch=rb['best_epoch'],
 all_primary_rows_labels_timestamps_identical=True,both_full_and_raw_input_certifications_passed=True,
 mean_mode_mass={n:certs[n]['uncertainty']['mean_mode_mass_general_public_historical'] for n in ids},
 uncertainty={n:certs[n]['uncertainty'] for n in ids},
 checkpoint_sha256={n:results[n]['model_sha256'] for n in ids},
 all_metrics_and_every_epoch_predictions_retained=True,
 limitations=['One seed; repeated adaptively reused Validation. Not confirmation.',
 'This control disables the historical branch; effective trainable capacity is not equal.',
 'Selected epochs are adaptive and can differ; fixed20 is the matched epoch mechanism comparison.',
 'Public route/history prototypes are latent hypotheses, not observed query trajectories or guaranteed drivable historical paths.',
 'TransferTraj has a different published processed cohort; no matched benchmark claim.'],
 verdict=verdict,goal_73_5_confirmation_achieved=False,Test='sealed',
 source_sha256=sha(__file__),actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(OUT/'conclusion_historica_fresca.json',result)
print(json.dumps(dict(treatment=result['treatment'],control=result['control'],fixed20_delta_s=fixed_comparison['delta_mae_s'],selected_delta_s=selected_comparison['delta_mae_s'],verdict=verdict)))
