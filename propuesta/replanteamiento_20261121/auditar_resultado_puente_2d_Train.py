"""Post-replaycalibration and originalgates; no modelchanges or label-dependent queryfilter."""
from base_replanteo import *
import torch
from puente_temporal_2d import score
run=OUT/'puente_temporal_2d_Train_s0';r=read(run/'resultado.json');proof=read(run/'reproduccion_independiente.json')
assert r['complete'] and proof['passed']
z=dict(np.load(run/'predicciones_held_Train.npz'));truth=z['true'];times=z['timestamp'];stats={}
for mode in ['none','linear','observed']:
 assert metric(truth,z[mode])==r['metrics'][mode]
 history=read(run/mode/'historia.json')
 assert len(history)==4 and all(row['train_n']==951718 and row['updates']==3718 for row in history)
 with np.load(run/mode/'predicciones_held_Train.npz') as arm:
  lp=arm['log_prob'].astype(np.float64);q=arm['quantiles']
 probability=np.exp(lp);normalization=float(np.max(np.abs(probability.sum(1)-1)));assert normalization<1e-5
 _,crps,nll=score(torch.tensor(lp),torch.tensor((truth/15).astype(np.int64)))
 for key,value in [('CRPS_s',float(crps.mean())),('bin_NLL',float(nll.mean()))]:assert abs(value-r['probability_scores'][mode][key])<1e-4
 descriptive=[]
 for lo,hi in [(60,300),(300,600),(600,900),(900,1800)]:
  mask=(truth>=lo)&(truth<hi);descriptive.append(dict(trueT_range_s=[lo,hi],metrics=metric(truth[mask],z[mode][mask])))
 stats[mode]=dict(metrics=r['metrics'][mode],probability_scores=r['probability_scores'][mode],
  mean_duration_entropy=float((-probability*lp).sum(1).mean()),
  mean_mass60s=float(probability[:,0].mean()),mean_mass1785s=float(probability[:,-1].mean()),
  archived_FP32_probability_normalization_maxerror=normalization,
  last_online_train_MAE_s=history[-1]['train_MAE_s'],
  held_minus_online_train_MAE_s=r['metrics'][mode]['mae_s']-history[-1]['train_MAE_s'],
  descriptive_trueT_groups=descriptive)
comparisons={}
for key,other,threshold in [('vs_neuralbaseline',z['none'],5),('vs_linearcontrol',z['linear'],2),('vs_CatBoostreference',z['reference'],5)]:
 p=paired(truth,z['observed'],other,times)
 assert abs(p['delta_mae_s']-r['comparisons'][key]['delta_mae_s'])<1e-12
 comparisons[key]=dict(delta_mae_s=p['delta_mae_s'],ci95_s=p['ci95_s'],required_improvement_s=threshold,
  passed=bool(p['delta_mae_s']<=-threshold and p['ci95_s'][1]<0))
passed=all(p['passed'] for p in comparisons.values());assert passed==r['passed']
a=dict(complete=True,practical_gate_passed=passed,statistics=stats,comparisons=comparisons,
 independent_probability_andraw65_replay_passed=True,
 main_gate_unchanged=True,secondary_matchedheadcontrol_pending=True,
 primary_best_unchanged_s=129.99003446901307,confirmation_goal_s=73.5,goal_achieved=False,
 limits=['Same10000reusedheldTrain and oneinitialization; not confirmation.',
 'Online trainingmetrics include dropout/changingparameters; train-held gaps descriptive.',
 'AuxiliarycontinuouspathNLL magnitudes are not directlycomparable between linear and observedtargets.',
 'SignedcorrelatedGPS motion is not nonnegativephysicalspeed, firstarrival, or a road-constrained simulator.',
 'GlobalStudent robustpathscale does not isolate independentGPSobservationerrors.',
 'TrueTgroups are retrospective diagnostics and never queryinputs or filters.'],
 all_results_retained_regardless131_06=True,retention_pending_secondary_control=True,
 Test='SEALED',Validation=False,source_sha256=sha(__file__))
write(run/'auditoria_resultado.json',a)
print(json.dumps(dict(complete=True,practical_gate_passed=passed,metrics=r['metrics'],secondary_control_pending=True)))
