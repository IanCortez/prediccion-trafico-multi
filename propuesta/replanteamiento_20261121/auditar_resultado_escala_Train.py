"""Post-replay audit of fullprecutoffTrain scale; no epoch/gate/label changes."""
from base_replanteo import *
run=OUT/'mezcla_individual_escala_Train_s0';r=read(run/'resultado.json');replay=read(run/'reproduccion_independiente.json')
assert r['complete'] and replay['passed'] and r['fixed_epoch']==4
assert sha(run/'predicciones_held_Train.npz')==r['predictions_sha256']
z=dict(np.load(run/'predicciones_held_Train.npz'));truth=z['true'];time=z['timestamp']
with np.load(OUT/'oraculo_tiempo_calle_train_s0/predicciones_held_Train.npz') as old:
 for key in ['row','timestamp','true']:assert np.array_equal(z[key],old[key])
stats={};groups={}
for mode in ['none','geometry','observed']:
 assert metric(truth,z[mode])==r['metrics'][mode]
 history=read(run/mode/'historia.json');assert len(history)==4 and all(v['train_n']==951718 and v['updates']==3718 for v in history)
 q=z[mode+'_quantiles'];assert q.shape==(10000,3) and np.isfinite(q).all()
 assert np.all(q[:,0]<=q[:,1]) and np.all(q[:,1]<=q[:,2])
 stats[mode]=dict(metrics=r['metrics'][mode],interval80=dict(coverage=float(np.mean((truth>=q[:,0])&(truth<=q[:,2]))),
  mean_width_s=float((q[:,2]-q[:,0]).mean()),median_width_s=float(np.median(q[:,2]-q[:,0]))),
  last_online_train_MAE_s=history[-1]['train_mae_s'],
  held_minus_online_train_s=r['metrics'][mode]['mae_s']-history[-1]['train_mae_s'])
 groups[mode]=[]
 for lo,hi in [(60,300),(300,600),(600,900),(900,1800)]:
  mask=(truth>=lo)&(truth<hi);groups[mode].append(dict(trueT_range_s=[lo,hi],metrics=metric(truth[mask],z[mode][mask])))
comparisons={}
for key,other,threshold in [('vs_neuralbaseline',z['none'],5),('vs_geometryproxy',z['geometry'],2),('vs_CatBoostreference',z['reference'],5)]:
 p=paired(truth,z['observed'],other,time);assert abs(p['delta_mae_s']-r['comparisons'][key]['delta_mae_s'])<1e-9
 comparisons[key]=dict(delta_mae_s=p['delta_mae_s'],ci95_s=p['ci95_s'],required_improvement_s=threshold,
  passed=p['delta_mae_s']<=-threshold and p['ci95_s'][1]<0)
passed=all(v['passed'] for v in comparisons.values());assert passed==r['passed']
old=read(OUT/'mezcla_individual_mediana_Train_s0/reproduccion_independiente.json')
scale={mode:r['metrics'][mode]['mae_s']-old['stable_metrics'][mode]['mae_s'] for mode in r['metrics']}
report=dict(complete=True,scope='Same10000reusedheldTrainqueries, now951718fit; no primaryValidation or confirmation.',
 practical_gate_passed=bool(passed),statistics=stats,comparisons=comparisons,change_vs_smallfit_held_MAE_s=scale,
 descriptive_trueT_groups=groups,original_rows_labels_and_cohort_preserved=True,independent_raw5_replay_passed=True,
 interpretation_limits=[
 'Dataamount, fitpopulation composition and stabletrainingnumerics change together from the smallpilot; not an isolated causal estimate of samplecount.',
 'Online trainingerror uses dropout and changing weights; the gap to finalheldprediction is descriptive, not identical-estimator risk.',
 'RepeatedTrainpilot use and one initialization; dayintervals exploratory, not independent confirmation.',
 'TrueTgroups are retrospective diagnostics, never queryinputs or a reason to change primaryfilters.'
 ],retention_decision_pending=True,
 next='Preregister fullunchangedprimary training onlyif the declaredtimegatepassed; otherwise critically reassess. Preserveallmetrics/predictions regardless131.06. No extraepochs/gate reduction.',
 goal_confirmation_s=73.5,primary_goal_proven=False,Test='SEALED',Validation=False,code_sha256=sha(__file__))
write(run/'auditoria_resultado.json',report)
print(json.dumps(dict(complete=True,practical_gate_passed=passed,metrics=r['metrics'],change_vs_smallfit_held_MAE_s=scale,primary_MAE_claim=False)))
