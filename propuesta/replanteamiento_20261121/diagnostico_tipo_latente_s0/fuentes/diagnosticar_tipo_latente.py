"""Train-only auxiliary call-type predictability; never an observed query feature."""
from base_replanteo import *
from features import features
from catboost import CatBoostClassifier
import shutil
access=deny_test()
run=OUT/'diagnostico_tipo_latente_s0'
assert not run.exists()
run.mkdir()
start=time.time()
d=dict(np.load(DATA/'train_x.npz'))
y=np.load(DATA/'train_y.npy')
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index']
ii=np.sort(np.random.default_rng(2026).choice(eligible,200000,replace=False))
part={k:v[ii] for k,v in d.items()}
X,names=features(part,enriched=False,calendar=True)
assert all(not n.startswith('call_') for n in names)
changed=dict(part);changed['call']=np.full(len(ii),'C')
Xi,ni=features(changed,enriched=False,calendar=True)
assert names==ni and np.array_equal(X,Xi),'Forbidden metadata must not alter query features'
labels=np.searchsorted(np.array(['A','B','C']),part['call'])
assert np.isin(part['call'],['A','B','C']).all()
fold=(part['timestamp']//(7*86400))%5
fit=fold!=0;held=fold==0
counts=np.bincount(labels[fit],minlength=3)
prior=counts/counts.sum()
model=CatBoostClassifier(iterations=800,depth=7,learning_rate=.07,l2_leaf_reg=5,loss_function='MultiClass',random_seed=0,thread_count=2,allow_writing_files=False,verbose=False,task_type='CPU')
model.fit(X[fit],labels[fit])
assert np.array_equal(model.classes_,[0,1,2])
p=model.predict_proba(X[held])
true=labels[held]
index=np.arange(len(true))
loss=-np.log(np.maximum(p[index,true],1e-12))
baseline=-np.log(prior[true])
delta=loss-baseline
days,di=np.unique(part['timestamp'][held]//86400,return_inverse=True)
sums=np.bincount(di,weights=delta);n=np.bincount(di)
boot=np.random.default_rng(1729).integers(len(n),size=(3000,len(n)))
ci=np.quantile(sums[boot].sum(1)/n[boot].sum(1),[.025,.975])
pred=p.argmax(-1)
confusion=np.zeros((3,3),np.int64);np.add.at(confusion,(true,pred),1)
description=[]
for k,c in enumerate(['A','B','C']):
    mask=d['call']==c
    description.append(dict(call=c,Train_n=int(mask.sum()),Train_mean_T_s=float(y[mask].mean()),Train_median_T_s=float(np.median(y[mask])),interpretation='Descriptive only; distance/context confounding, not a causal type effect'))
np.savez_compressed(run/'held_Train_predicciones.npz',row=part['row'][held],timestamp=part['timestamp'][held],true_call=true,predicted_probabilities=p.astype(np.float32))
informative=bool(delta.mean()<-.05 and ci[1]<0)
if informative:model.save_model(str(run/'clasificador.cbm'))
(run/'fuentes').mkdir()
shutil.copy2(OUT/'diagnosticar_tipo_latente.py',run/'fuentes/diagnosticar_tipo_latente.py')
result=dict(scope='Auxiliary metadata classification on held Train weeks only, NOT travel-time MAE and NOT a primary benchmark result',features=names,query_information=['o_lon','o_lat','d_lon','d_lat','timestamp'],actual_call_is_not_a_query_input=True,metadata_intervention_feature_invariance=True,fit_n=int(fit.sum()),held_n=int(held.sum()),held_rule='Train departure UTCweek modulo5==0; deterministic OD/calendar features only, no fitted target-history feature banks',iterations_fixed_before_held=True,held_Train_classification=dict(NLL=float(loss.mean()),constant_prior_NLL=float(baseline.mean()),delta_NLL=float(delta.mean()),CI95_by_Train_day=ci,accuracy=float((pred==true).mean()),majority_accuracy=float((true==prior.argmax()).mean()),confusion_A_B_C=confusion),informative_auxiliary_label=informative,Train_description=description,interpretation='Predictability can justify investigating an auxiliary semantic label. It does not prove extra information beyond a strong neural representation or any travel-time MAE gain. A future latent-type model must marginalize predicted type at query time; observed CALL_TYPE remains ineligible as a primary query feature.',historical_enriched_control='od_mas_call_mae_d10_s0 used observed metadata and is not eligible. This diagnostic is not a repeat of that information contract.',no_GPU=True,Test_access=False,actual_files_accessed=sorted(access),code_sha256=sha(__file__),seconds=time.time()-start)
write(run/'resultado.json',result)
print(json.dumps({k:result[k] for k in ['held_Train_classification','informative_auxiliary_label','Train_description','seconds']},default=lambda x:x.tolist()),flush=True)
