"""Global semanticroute posterior versus time-only and matchedrandomcode controls, Train only."""
from base_replanteo import *
import pandas as pd,shutil,gc
from numba import set_num_threads
from catboost import CatBoostClassifier,CatBoostRegressor
from inferencia import feature_input
from distribucion_cuantiles_ruta import mixture_quantiles,rearrange
access=deny_test();start=time.time();set_num_threads(2)
run=OUT/'piloto_posterior_ruta_global_train_s0';run.mkdir(exist_ok=True)
sources=['piloto_posterior_ruta_global_train.py','distribucion_cuantiles_ruta.py','base_replanteo.py']
if not (run/'fuentes').exists():
 (run/'fuentes').mkdir()
 for name in sources:shutil.copy2(OUT/name,run/'fuentes'/name)
else:
 for name in sources:assert sha(OUT/name)==sha(run/'fuentes'/name),'Sources immutable during resume'
assert not (run/'resultado.json').exists()
codefolder=OUT/'codigos_ruta_global_train_s0';codemeta=read(codefolder/'resultado.json');assert codemeta['complete']
assert sha(codefolder/'codigos_y_firmas_Train.npz')==codemeta['outputs_sha256']
plan=dict(scope='Chronological40kfit/10kheldTrain raw5ODgate, not primaryValidation or confirmation.',
 hypothesis='Coherentwhole-routecategory may transfer moretimeinformation than independentlocalgreedychoices.',
 latent='p(T,C|x)=p(C|x)*p(T|x,C). Codeconstructorfit-only geometryresidual PCA12/KMeans16, noT incodeconstruction; C actualTrainauxlabel only. Inferencemarginalizes allcodes without actualqueryroute.',
 input='Same206publicODfeatures plus90publicshorteststreet-signature features forEVERYbranch; no actualheldroute,N,timing,driver.',
 branches=['public_time_only_quantiles','semantic_global_route_marginal','matched_random_route_marginal'],
 matched_control='Randompermutation of truefitcodes keepsclassfrequencies identical; sameclassifier andconditionaltimeparams/capacity. Also directtime-onlyquantilecontrol.',
 classifier=dict(iterations=300,depth=6,learning_rate=.05,l2_leaf_reg=5,loss_function='MultiClass',random_seed=20261122,thread_count=2,allow_writing_files=False,verbose=False),
 conditional_time=dict(iterations=1000,depth=7,learning_rate=.05,l2_leaf_reg=5,loss_function='MultiQuantile:alpha=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9',
  random_seed=20261122,thread_count=2,allow_writing_files=False,verbose=False),
 categorical=dict(max_ctr_complexity=1,ctr_leaf_count_limit=200000),
 seed=20261122,fit_n=40000,held_n=10000,checkpoint='Fixedfinaltrees, no heldstopping/parametersweep.',
 distribution='Monotonerearrange9conditionalquantiles clipped60..1785; linearlyinterpolatedCDF betweenquantiles withfixedsupporttails. SumCDFsweightedbyclassposterior, generalizedinversevia36bisections. Continuousseconds pointforecast, originalT unchanged.',
 semantic_gate='>=2seconds gainoverstored206CatBoostMAEcontrol withupperdayCI<0 AND>=1second gainovermatchedrandom withupperCI<0 AND no worse thanpublictime-onlyquantilecontrol.',
 public_only_gate='Independentlypreregistered: >=2seconds gainoverstored206CatBoostMAEcontrol withupperdayCI<0. Ifonlythispasses, teachersemanticsnotcredited andonlypublicsignaturetimebranchadvances.',
 selection='Semantic ifitsgatepasses, otherwisepublic-only ifitsgatepasses, else nofullprimaryexpansion. Randomcontrol neverselected.',
 controls='AllforecastscomputedbeforeheldTTmetrics. Actualheldcode used onlyafterforecasts forprivilegedoracle/classaccuracy diagnostics.',
 resources='CPU2, <=150MiB; resumefinalstageweights byverifiedmanifest, noCDFondisk.',Validation=False,Test=False)
if (run/'plan.json').exists():assert read(run/'plan.json')==plan
else:write(run/'plan.json',plan)
source=OUT/'oraculo_tiempo_calle_train_s0'
with np.load(source/'auxiliares_Train.npz') as z:q={key:z[key] for key in ['row','train_index','timestamp','true']}
codes=dict(np.load(codefolder/'codigos_y_firmas_Train.npz'))
for key in ['row','train_index','timestamp']:assert np.array_equal(codes[key],q[key])
nf=40000;true_code=codes['code'][:nf].astype(np.int32);random_code=true_code[np.random.default_rng(20261122).permutation(nf)]
assert np.array_equal(np.bincount(true_code),np.bincount(random_code))
d,X,names=feature_input('train',read(BEST/'resultado.json'));keep=np.asarray([not n.startswith('mm_') for n in names]);assert keep.sum()==206
assert np.array_equal(d['row'][q['train_index']],q['row'])
X=np.column_stack([X[q['train_index']][:,keep],codes['public_signature'].astype(np.float32)]);assert X.shape==(50000,296) and np.isfinite(X).all()
y=q['true'].astype(np.float64);fitX=X[:nf];heldX=X[nf:];feature_names=[n for n,k in zip(names,keep) if k]+['public_street_signature_'+str(i) for i in range(90)]
classes=np.unique(true_code);assert np.array_equal(classes,np.unique(random_code))
file=run/'etapas_completas.json';manifest=read(file) if file.exists() else {}
def fit_stage(name,kind,label=None):
 path=run/(name+'.cbm')
 cls=CatBoostClassifier if kind=='classifier' else CatBoostRegressor
 if name in manifest:
  assert sha(path)==manifest[name]['sha256'];model=cls();model.load_model(str(path));return model
 assert not path.exists(),'Uncommittedstagefile requires inspection, not automaticrerun'
 print(json.dumps(dict(stage='fit_'+name,seconds=time.time()-start)),flush=True)
 if kind=='classifier':
  model=cls(**plan['classifier']);model.fit(fitX,label)
 else:
  params=dict(plan['conditional_time'])
  if label is not None:params.update(plan['categorical'])
  model=cls(**params)
  if label is None:model.fit(fitX,y[:nf])
  else:
   frame=pd.DataFrame(fitX,columns=feature_names);frame['latent_route_code']=label.astype(str)
   model.fit(frame,y[:nf],cat_features=['latent_route_code'])
 model.save_model(str(path));manifest[name]=dict(file=path.name,sha256=sha(path),bytes=path.stat().st_size,kind=kind);write(file,manifest);return model
# APIpreflight usesfitlabels only, twoiterations, neverseenheldTT. No scientificcandidate claim.
small=pd.DataFrame(fitX[:512],columns=feature_names);small['latent_route_code']=true_code[:512].astype(str)
pre=CatBoostRegressor(**dict(plan['conditional_time'],iterations=2));pre.fit(small,y[:512],cat_features=['latent_route_code']);preq=pre.predict(small.iloc[:17]);assert preq.shape==(17,9) and np.isfinite(preq).all();del pre,small
write(run/'verificacion_API.json',dict(passed=True,MultiQuantile_categorical_supported=True,fit_only=True))
plain=fit_stage('public_time_only','time');plain_q=rearrange(plain.predict(heldX));plain_forecast=mixture_quantiles(plain_q[:,None],np.ones((len(heldX),1)))
del plain;gc.collect()
outputs={};class_probs={};conditional_saved={}
def forecast(classifier,conditional,public_query):
 probability=classifier.predict_proba(public_query);labels=np.asarray(classifier.classes_,np.int32);quantiles=[]
 for label in labels:
  frame=pd.DataFrame(public_query,columns=feature_names);frame['latent_route_code']=str(int(label))
  quantiles.append(rearrange(conditional.predict(frame)))
 component=np.stack(quantiles,axis=1)
 assert probability.shape==component.shape[:2] and np.isfinite(probability).all() and np.allclose(probability.sum(1),1)
 return mixture_quantiles(component,probability),probability,component
for branch,label in [('semantic',true_code),('random',random_code)]:
 classifier=fit_stage(branch+'_posterior','classifier',label);conditional=fit_stage(branch+'_conditional_time','time',label)
 forecast_q,probability,component=forecast(classifier,conditional,heldX)
 # The methodacceptsonlypublicqueryfeatures; no actualheldcode/route/length/T argument.
 probe=17;reverse,_,_=forecast(classifier,conditional,heldX[:probe][::-1]);single,_,_=forecast(classifier,conditional,heldX[:1])
 assert np.array_equal(forecast_q[:probe],reverse[::-1]) and np.array_equal(forecast_q[:1],single)
 outputs[branch]=forecast_q;class_probs[branch]=probability;conditional_saved[branch]=component
 if branch=='semantic':
  predicted_codes=np.asarray(classifier.classes_)[probability.argmax(1)]
  # PrivilegeddiagnosticonlyAFTER allsemanticpublicforecasts, never usedbyforecast().
  actual=pd.DataFrame(heldX,columns=feature_names);actual['latent_route_code']=codes['code'][nf:].astype(str)
  oracle_q=rearrange(conditional.predict(actual));diagnostic_code_accuracy=float(np.mean(predicted_codes==codes['code'][nf:]))
 del classifier,conditional;gc.collect()
# Scoring starts onlyafter everyfixedbranch forecast exists; no model adapts to these metrics.
ref=dict(np.load(source/'predicciones_held_Train.npz'));oraclemeta=read(source/'resultado.json')
for key in ['row','timestamp']:assert np.array_equal(ref[key],q[key][nf:])
assert np.array_equal(ref['true'],y[nf:]) and metric(y[nf:],ref['control'])==oraclemeta['metrics']['control']
m={name:metric(y[nf:],pred[:,1]) for name,pred in [('public_only',plain_forecast),('semantic',outputs['semantic']),('random',outputs['random'])]}
comparisons=dict(public_vs_reference=paired(y[nf:],plain_forecast[:,1],ref['control'],q['timestamp'][nf:]),
 semantic_vs_reference=paired(y[nf:],outputs['semantic'][:,1],ref['control'],q['timestamp'][nf:]),
 semantic_vs_random=paired(y[nf:],outputs['semantic'][:,1],outputs['random'][:,1],q['timestamp'][nf:]),
 semantic_vs_public=paired(y[nf:],outputs['semantic'][:,1],plain_forecast[:,1],q['timestamp'][nf:]))
for value in comparisons.values():value['interpretation']='ChronologicalheldTrain, fixedfinalbranchmodels; repeatedgates exploratory, not primaryValidation/confirmation.'
good=lambda value,minimum:value['delta_mae_s']<=-minimum and value['ci95_s'][1]<0
semantic_passed=bool(good(comparisons['semantic_vs_reference'],2) and good(comparisons['semantic_vs_random'],1) and m['semantic']['mae_s']<=m['public_only']['mae_s'])
public_passed=bool(good(comparisons['public_vs_reference'],2));selected='semantic' if semantic_passed else 'public_only' if public_passed else None
np.savez_compressed(run/'predicciones_held_Train.npz',row=q['row'][nf:],timestamp=q['timestamp'][nf:],true=y[nf:],public_only=plain_forecast[:,1],public_quantiles=plain_forecast,
 semantic=outputs['semantic'][:,1],semantic_quantiles=outputs['semantic'],random=outputs['random'][:,1],random_quantiles=outputs['random'],
 reference=ref['control'],privileged_true_code=oracle_q[:,4],semantic_code_probability=class_probs['semantic'],random_code_probability=class_probs['random'])
result=dict(passed=selected is not None,selected=selected,semantic_gate_passed=semantic_passed,public_only_gate_passed=public_passed,metrics=m,reference=metric(y[nf:],ref['control']),comparisons=comparisons,
 privileged_true_code_oracle=metric(y[nf:],oracle_q[:,4]),held_code_accuracy_diagnostic=diagnostic_code_accuracy,classes=classes,
 actualheldroutecode_not_public_input=True,batch_reverse_singleton_allquantiles_exact=True,all_query_labels_preserved=True,
 source_sha256=sha(run/'fuentes/piloto_posterior_ruta_global_train.py'),distribution_source_sha256=sha(run/'fuentes/distribucion_cuantiles_ruta.py'),
 codebook_manifest_sha256=sha(codefolder/'resultado.json'),predictions_sha256=sha(run/'predicciones_held_Train.npz'),fitted_stages=manifest,
 Validation_access=False,Test_access=False,seconds=time.time()-start,scope=plan['scope'])
write(run/'resultado.json',result)
keep_models={'semantic_posterior','semantic_conditional_time'} if selected=='semantic' else {'public_time_only'} if selected=='public_only' else set()
removed=[]
for name,entry in manifest.items():
 if name in keep_models:continue
 path=run/entry['file'];assert path.resolve().is_relative_to(run.resolve()) and sha(path)==entry['sha256'];removed.append(entry);path.unlink()
write(run/'pesos_descartados.json',dict(removed=removed,selected_models=sorted(keep_models),all_completed_predictions_preserved=True))
assert sum(p.stat().st_size for p in run.rglob('*') if p.is_file())<150*1024**2
print(json.dumps(dict(passed=result['passed'],selected=selected,metrics=m,reference=result['reference'],privileged_code=result['privileged_true_code_oracle'],seconds=result['seconds'])),flush=True)
