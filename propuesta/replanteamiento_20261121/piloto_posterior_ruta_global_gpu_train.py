"""Efficient scalarquantile implementation of the sameglobalroute contrasts, before observingoldheldscores."""
from base_replanteo import *
import pandas as pd,shutil,gc
from numba import set_num_threads
from catboost import CatBoostClassifier,CatBoostRegressor
from inferencia import feature_input
from distribucion_cuantiles_ruta import mixture_quantiles,rearrange
access=deny_test();start=time.time();set_num_threads(2);assert read(OUT/'verificacion_catboost_cuantiles_gpu.json')['passed']
run=OUT/'piloto_posterior_ruta_global_gpu_train_s0';run.mkdir(exist_ok=True);sources=['piloto_posterior_ruta_global_gpu_train.py','distribucion_cuantiles_ruta.py','base_replanteo.py']
if not (run/'fuentes').exists():
 (run/'fuentes').mkdir()
 for name in sources:shutil.copy2(OUT/name,run/'fuentes'/name)
else:
 for name in sources:assert sha(OUT/name)==sha(run/'fuentes'/name)
assert not (run/'resultado.json').exists()
common=dict(random_seed=20261122,thread_count=2,task_type='GPU',devices='0',gpu_ram_part=.25,one_hot_max_size=32,border_count=128,allow_writing_files=False,verbose=False)
plan=dict(scope='Chronological40kfit/10kheldTrain raw5ODgate, not primaryValidation/confirmation.',
 reason='CPUsharedMultiQuantilefirst1000treestage took>17minutes withoutcompletion, verifiedliveCPU. Use9scalarGPUquantileregressors withsamebranches/data beforeanyoldheldscoreobserved.',
 implementation_difference='Nineindependenttreesets instead ofonesharedmultioutputtree. Everybranch hasidenticalscalarquantilecapacity; semantic/random same17wayclassifiercapacity. Noclaimbit-identicaltoCPUtraining.',
 training_target='OriginalTseconds/600 reversibleunitnormalizationforGPUoptimization; multiplyallforecastsby600beforeevaluation, originallabels unchanged.',
 common_GPU=common,time=dict(iterations=1000,depth=7,learning_rate=.05,l2_leaf_reg=5),
 classifier=dict(iterations=300,depth=6,learning_rate=.05,l2_leaf_reg=5,loss_function='MultiClass'),
 alphas=[.1,.2,.3,.4,.5,.6,.7,.8,.9],input='Same206publicfeatures+90publicshortestsignature; actualroutecodeTrainteacheronly.',
 branches=['public_only','semantic','random'],random_control='Exactclassfrequencypreservingfitlabelpermutationseed20261122.',
 categorical='17latentcodesonehot, max32; no highcardinalitytargetCTR.',stopping='Fixedfinal1000trees/scalarquantile and300classifier; heldscoredonlyafterallbranches.',
 semantic_gate='>=2secondswithupperCI<0overstoredreference AND>=1secondwithupperCI<0overmatchedrandom ANDno worse thanpublic-only.',
 public_gate='Independentlypreregistered>=2secondswithupperCI<0overstoredreference; no semanticclaimifonlythispasses.',
 distribution='Samefixed9quantilemonotonerearrangement, continuouspiecewiselinearCDF mixture and36bisectioninversion, support60..1785.',
 replay='GPUtrainingmaybenondeterministic; savedmodelCPUinference must replayexactly. CPUforecastthread_count2 explicit.',
 resources='OneGPUfraction.25, CPU2,<=150MiB; resumeverifiedfinalmodelsbySHA, noCDFcache.',Validation=False,Test=False)
if (run/'plan.json').exists():assert read(run/'plan.json')==plan
else:write(run/'plan.json',plan)
codefolder=OUT/'codigos_ruta_global_train_s0';cm=read(codefolder/'resultado.json');assert sha(codefolder/'codigos_y_firmas_Train.npz')==cm['outputs_sha256'];codes=dict(np.load(codefolder/'codigos_y_firmas_Train.npz'))
source=OUT/'oraculo_tiempo_calle_train_s0'
with np.load(source/'auxiliares_Train.npz') as z:q={key:z[key] for key in ['row','train_index','timestamp','true']}
for key in ['row','train_index','timestamp']:assert np.array_equal(q[key],codes[key])
d,X,names=feature_input('train',read(BEST/'resultado.json'));keep=np.asarray([not n.startswith('mm_') for n in names]);assert keep.sum()==206
X=np.column_stack([X[q['train_index']][:,keep],codes['public_signature'].astype(np.float32)]);nf=40000;fitX=X[:nf];heldX=X[nf:];y=q['true'].astype(np.float64)
fnames=[n for n,k in zip(names,keep) if k]+['public_street_signature_'+str(i) for i in range(90)]
truecode=codes['code'][:nf].astype(np.int32);randomcode=truecode[np.random.default_rng(20261122).permutation(nf)]
assert np.array_equal(np.bincount(truecode),np.bincount(randomcode))
file=run/'etapas_completas.json';manifest=read(file) if file.exists() else {}
def fit_model(name,kind,label=None,alpha=None):
 path=run/(name+'.cbm');cls=CatBoostClassifier if kind=='classifier' else CatBoostRegressor
 if name in manifest:
  assert sha(path)==manifest[name]['sha256'];m=cls();m.load_model(str(path));return m
 assert not path.exists(),'Uncommittedmodelrequiresinspection'
 params=dict(common);params.update(plan['classifier'] if kind=='classifier' else plan['time'])
 if kind!='classifier':params['loss_function']='Quantile:alpha='+str(alpha)
 m=cls(**params)
 if label is None:frame=fitX
 else:
  frame=pd.DataFrame(fitX,columns=fnames);frame['latent_route_code']=label.astype(str)
 if kind=='classifier':m.fit(fitX,label)
 elif label is None:m.fit(frame,y[:nf]/600)
 else:m.fit(frame,y[:nf]/600,cat_features=['latent_route_code'])
 m.save_model(str(path));manifest[name]=dict(file=path.name,sha256=sha(path),bytes=path.stat().st_size,kind=kind,alpha=alpha);write(file,manifest)
 print(json.dumps(dict(stage='complete_'+name,seconds=time.time()-start)),flush=True);return m
def fit_quantiles(prefix,label=None):
 for j,alpha in enumerate(plan['alphas']):
  m=fit_model(prefix+'_q'+str(j+1),'time',label,alpha);del m;gc.collect()
def load_models(prefix):
 out=[]
 for j in range(9):
  name=prefix+'_q'+str(j+1);m=CatBoostRegressor();m.load_model(str(run/manifest[name]['file']));out.append(m)
 return out
fit_quantiles('public_only');plain_models=load_models('public_only')
plain_q=rearrange(np.column_stack([m.predict(heldX,thread_count=2)*600 for m in plain_models]))
plain_forecast=mixture_quantiles(plain_q[:,None],np.ones((len(heldX),1)));del plain_models;gc.collect()
outputs={};probs={}
def forecast(classifier,models,query):
 probability=classifier.predict_proba(query,thread_count=2);classes=np.asarray(classifier.classes_,np.int32);parts=[]
 for label in classes:
  frame=pd.DataFrame(query,columns=fnames);frame['latent_route_code']=str(int(label))
  parts.append(rearrange(np.column_stack([m.predict(frame,thread_count=2)*600 for m in models])))
 component=np.stack(parts,axis=1);assert probability.shape==component.shape[:2] and np.allclose(probability.sum(1),1)
 return mixture_quantiles(component,probability),probability
for branch,label in [('semantic',truecode),('random',randomcode)]:
 classifier=fit_model(branch+'_posterior','classifier',label)
 fit_quantiles(branch,label);models=load_models(branch)
 f,prob=forecast(classifier,models,heldX);r,_=forecast(classifier,models,heldX[:17][::-1]);s,_=forecast(classifier,models,heldX[:1])
 assert np.array_equal(f[:17],r[::-1]) and np.array_equal(f[:1],s);outputs[branch]=f;probs[branch]=prob
 if branch=='semantic':
  predicted_code=np.asarray(classifier.classes_)[prob.argmax(1)]
  frame=pd.DataFrame(heldX,columns=fnames);frame['latent_route_code']=codes['code'][nf:].astype(str)
  oracle_q=rearrange(np.column_stack([m.predict(frame,thread_count=2)*600 for m in models]))
 del classifier,models;gc.collect()
ref=dict(np.load(source/'predicciones_held_Train.npz'))
for key in ['row','timestamp']:assert np.array_equal(ref[key],q[key][nf:])
assert np.array_equal(ref['true'],y[nf:])
metrics={name:metric(y[nf:],f[:,1]) for name,f in [('public_only',plain_forecast),('semantic',outputs['semantic']),('random',outputs['random'])]}
comparison=dict(public_vs_reference=paired(y[nf:],plain_forecast[:,1],ref['control'],q['timestamp'][nf:]),
 semantic_vs_reference=paired(y[nf:],outputs['semantic'][:,1],ref['control'],q['timestamp'][nf:]),
 semantic_vs_random=paired(y[nf:],outputs['semantic'][:,1],outputs['random'][:,1],q['timestamp'][nf:]),
 semantic_vs_public=paired(y[nf:],outputs['semantic'][:,1],plain_forecast[:,1],q['timestamp'][nf:]))
for v in comparison.values():v['interpretation']='FixedfinalscalarGPUquantiles, chronologicalheldTrain, not primaryValidation/confirmation.'
good=lambda v,minimum:v['delta_mae_s']<=-minimum and v['ci95_s'][1]<0
semantic_passed=bool(good(comparison['semantic_vs_reference'],2) and good(comparison['semantic_vs_random'],1) and metrics['semantic']['mae_s']<=metrics['public_only']['mae_s'])
plain_passed=bool(good(comparison['public_vs_reference'],2));selected='semantic' if semantic_passed else 'public_only' if plain_passed else None
np.savez_compressed(run/'predicciones_held_Train.npz',row=q['row'][nf:],timestamp=q['timestamp'][nf:],true=y[nf:],reference=ref['control'],
 public_only=plain_forecast[:,1],public_quantiles=plain_forecast,semantic=outputs['semantic'][:,1],semantic_quantiles=outputs['semantic'],
 random=outputs['random'][:,1],random_quantiles=outputs['random'],semantic_code_probability=probs['semantic'],random_code_probability=probs['random'],privileged_true_code=oracle_q[:,4])
result=dict(passed=selected is not None,selected=selected,semantic_gate_passed=semantic_passed,public_gate_passed=plain_passed,metrics=metrics,reference=metric(y[nf:],ref['control']),comparisons=comparison,
 privileged_true_code_oracle=metric(y[nf:],oracle_q[:,4]),held_code_accuracy_diagnostic=float(np.mean(predicted_code==codes['code'][nf:])),
 original_labels_evaluated_in_seconds=True,noactualheldroutecode_in_public_forecast=True,reverse_singleton_allquantiles_exact=True,
 sources={name:sha(run/'fuentes'/name) for name in sources},codebook_manifest_sha256=sha(codefolder/'resultado.json'),fitted_stages=manifest,
 predictions_sha256=sha(run/'predicciones_held_Train.npz'),Validation_access=False,Test_access=False,seconds=time.time()-start,scope=plan['scope'])
write(run/'resultado.json',result);keep_prefix=selected;removed=[]
for name,entry in manifest.items():
 keep=keep_prefix and name.startswith(keep_prefix+'_')
 if keep:continue
 path=run/entry['file'];assert path.resolve().is_relative_to(run.resolve()) and sha(path)==entry['sha256'];removed.append(entry);path.unlink()
write(run/'pesos_descartados.json',dict(removed=removed,selected=selected,all_completed_predictions_preserved=True))
assert sum(p.stat().st_size for p in run.rglob('*') if p.is_file())<150*1024**2
print(json.dumps(dict(passed=result['passed'],selected=selected,metrics=metrics,reference=result['reference'],privileged_code=result['privileged_true_code_oracle'],seconds=result['seconds'])),flush=True)
