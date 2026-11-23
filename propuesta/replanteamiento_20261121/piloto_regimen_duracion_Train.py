"""CPU-only chronological heldTrain gate for a new duration-regime factorization."""
from base_replanteo import *
import shutil,joblib,gc
from sklearn.metrics import roc_auc_score,log_loss,brier_score_loss
from threadpoolctl import threadpool_limits
from inferencia import feature_input
from bosque_distribucional_regimen import LeafDurationForest,DurationRegimeForest
access=deny_test();start=time.time();run=OUT/'piloto_regimen_duracion_Train_s0';assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
sources=['piloto_regimen_duracion_Train.py','bosque_distribucional_regimen.py','base_replanteo.py']
for name in sources:shutil.copy2(OUT/name,run/'fuentes'/name)
plan=dict(scope='Reusedchronological40kfit/10kheldTrain signalgate, not primaryValidation, confirmation or a targetcohortchange.',
 motivation='Descriptiveprimaryaudit found9.9percentofqueriescontribute40percentofabsoluteerror. Study identifiable supervisedtime-regimes ratherthan anotherroutegenerator.',
 boundary_s=900,query='Same206publicfeatures, excludingall10mm_profiles to prevent heldperiodlabel leakage.',
 density='116original15secondbins, empiricalleafdistributions withoneTrainmarginalpseudoobservationperleaf.',
 factorization='P(T|OD)=P(short|OD)*P(T|short,OD)+P(long|OD)*P(T|long,OD); trueTregime isTrainlabel only, neverheldinput.',
 global_control='256ExtraTrees depth14,minleaf20,maxfeatures1,seed20261122,CPU2',
 treatment='128binarygate trees+128conditionaldensitytrees perregime, same depth/minleaf. Equal approximate fittedrow-tree visits; capacitynotexactidentical.',
 gate='>=2seconds betterthan stored133.355393reference AND>=1second betterthan globaldensitycontrol; exploratorydayCIupper<0forboth.',
 no_tuning='Onefixedboundary, treecount/depth/smoothing. No held stopping, no laterboundaries selected.',
 oracle='No targetregime oracle presented as a publicforecast.',
 resources='CPU2 only whileoneGPUjointmodeltrains; save qualifiedpilotweights onlyifgatepasses, allpredictions/metrics/sourcesalways.',
 comparison_limitation='Pilotrows inheritedfrompreviousrouteinstrument and repeatedlyexamined; no independentgeneralization or primaryMAEclaim.',
 Validation=False,Test=False)
write(run/'plan.json',plan)
source=OUT/'oraculo_tiempo_calle_train_s0';meta=read(source/'resultado.json')
assert sha(source/'auxiliares_Train.npz')==meta['auxiliary_sha256']
with np.load(source/'auxiliares_Train.npz') as z:q={key:z[key] for key in ['row','train_index','timestamp','true']}
d,X,names=feature_input('train',read(BEST/'resultado.json'));keep=np.array([not name.startswith('mm_') for name in names]);assert keep.sum()==206
X=X[q['train_index']][:,keep];assert len(X)==50000
y=np.load(DATA/'train_y.npy')[q['train_index']]
assert np.array_equal(y,q['true']) and np.array_equal(q['row'],d['row'][q['train_index']])
assert np.max(q['timestamp'][:40000]+y[:40000])<np.min(q['timestamp'][40000:])
fitX=X[:40000];heldX=X[40000:];fit_y=y[:40000];truth=y[40000:]
reference=dict(np.load(source/'predicciones_held_Train.npz'))
for key in ['row','timestamp']:assert np.array_equal(reference[key],q[key][40000:])
assert np.array_equal(reference['true'],truth)
with threadpool_limits(limits=2):
 tick=time.time();control=LeafDurationForest().fit(fitX,fit_y);cq=control.predict_quantiles(heldX)
 print(json.dumps(dict(stage='global_density_complete',seconds=time.time()-tick)),flush=True)
 tick=time.time();model=DurationRegimeForest().fit(fitX,fit_y);tq,gate=model.predict_quantiles(heldX)
 print(json.dumps(dict(stage='regime_density_complete',seconds=time.time()-tick)),flush=True)
 reverse,rg=model.predict_quantiles(heldX[:17][::-1]);single,sg=model.predict_quantiles(heldX[:1])
 assert np.array_equal(tq[:17],reverse[::-1]) and np.array_equal(tq[:1],single)
 assert np.array_equal(gate[:17],rg[::-1]) and np.array_equal(gate[:1],sg)
global_m=metric(truth,cq[:,1]);m=metric(truth,tq[:,1]);ref_m=metric(truth,reference['control'])
comparisons=dict(vs_global=paired(truth,tq[:,1],cq[:,1],q['timestamp'][40000:]),vs_reference=paired(truth,tq[:,1],reference['control'],q['timestamp'][40000:]))
for item in comparisons.values():item['interpretation']='ExploratoryreusedchronologicalheldTrain, no primaryValidation orconfirmation.'
passed=all(comparisons[key]['delta_mae_s']<=-amount and comparisons[key]['ci95_s'][1]<0 for key,amount in [('vs_global',1),('vs_reference',2)])
np.savez_compressed(run/'predicciones_held_Train.npz',row=q['row'][40000:],timestamp=q['timestamp'][40000:],true=truth,
 control=cq[:,1],control_quantiles=cq,treatment=tq[:,1],treatment_quantiles=tq,probability_long=gate,reference=reference['control'])
weights=None
if passed:
 path=run/'bosque_regimen.joblib';joblib.dump(dict(model=model,features=[n for n,k in zip(names,keep) if k],plan=plan),path,compress=3)
 assert path.stat().st_size<250*1024**2;weights=dict(file=str(path),sha256=sha(path),MiB=path.stat().st_size/1024**2)
regime=(truth>=900).astype(int)
result=dict(complete=True,passed=bool(passed),metrics=m,global_density_control=global_m,reference=ref_m,comparisons=comparisons,
 binary_gate=dict(AUC=float(roc_auc_score(regime,gate)),log_loss=float(log_loss(regime,gate)),brier=float(brier_score_loss(regime,gate)),
  true_long_fraction=float(regime.mean()),mean_probability_long=float(gate.mean())),
 uncertainty=dict(coverage80=float(np.mean((truth>=tq[:,0])&(truth<=tq[:,2]))),width80_s=float((tq[:,2]-tq[:,0]).mean())),
 originalT_seconds=True,no_held_groundtruth_regime_used=True,reverse_singleton_quantiles_gate_exact=True,
 sources={name:sha(run/'fuentes'/name) for name in sources},predictions_sha256=sha(run/'predicciones_held_Train.npz'),weights=weights,
 future_if_pass='Testfulloriginalprimarycohort withstructuredconditionaldensity expert; no inference-regimeoracle, no ordinarypointaverage in placeofmixturemedian.',
 future_if_fail='RetirethisCPUforestfactorization; no boundary/treecount sweeps, nofullprimarytrial basedsolelyon AUC.',
 Validation_access=False,Test_access=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(run/'resultado.json',result);print(json.dumps(read(run/'resultado.json')),flush=True)
