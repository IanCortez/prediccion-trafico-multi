"""Chronological Train-only historicaltime signal after corrected orderedsource discovery."""
from base_replanteo import *
import gc,shutil
from catboost import CatBoostRegressor
from inferencia import feature_input
from recuperar_fuentes_ordenadas import OrderedSourceMemory
from features_cdf_fuentes import source_features,GEOM_NAMES,GRID
access=deny_test();start=time.time();run=OUT/'piloto_cdf_ordenada_Train_s0';assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
proof=read(OUT/'verificacion_fuentes_ordenadas_v2_s0/resultado.json');assert proof['passed']
assert proof['sources']['recuperar_fuentes_ordenadas.py']==sha(OUT/'recuperar_fuentes_ordenadas.py')
sources=['piloto_cdf_ordenada_Train.py','recuperar_fuentes_ordenadas.py','features_cdf_fuentes.py','base_replanteo.py']
for name in sources:shutil.copy2(OUT/name,run/'fuentes'/name)
plan=dict(scope='Reused chronological40000fit/10000heldTrain timeprediction gate; no primaryValidation or confirmation.',
 hypothesis='Ordered parent-based retrieval supplies historical interval times missing from independently truncated endpointlists.',
 sources='Original immutable241.76MiB pointbank only, arrival before heldcutoff. Trainqueries exclude whole parentUTCweekfold BEFORE tree construction; helduses all earlierparents.',
 representation='15geometry/calendar/boundary summaries plus3boundary-group weightedCDFs on116fixedoriginaltimebins. Full/touch/internal source groups, same64sources andK2048.',
 geometrycontrol='SameCDFdimension, groups, geometry and weights; replace observed historicaltime with sourcechord/(25kmh)*distance-adjustment. Different effective featureentropy is disclosed; not identical informationcontent.',
 no_query_targets='Query API has raw5fields only. No backbone pretrained on heldTrainlabels. Actualheld route/N/T never features.',
 learner=dict(iterations=1000,depth=8,learning_rate=.05,l2_leaf_reg=5,loss='MAE',seed=20261124,target='Originalseconds/600 and exactinverse600'),
 gate='>=5sbetterthan freshpublic206baseline ANDstored133.355393reference; >=2sbetterthan geometryCDFcontrol, exploratorydayCIupper<0 for all.',
 no_tuning='One fixedCDFgrid/groups/gates/treeconfig. Failedtimegate is not repaired by changing K/radius/binning or selecting an epoch.',
 resources='OneGPU .25cap, CPU2 for sourceconstruction. No extraGPSbank; compactsourceindices <=30MiB, allmetrics/predictions always preserved.',
 validation=False,Test=False)
write(run/'plan.json',plan)
source=OUT/'oraculo_tiempo_calle_train_s0'
assert sha(source/'auxiliares_Train.npz')==read(source/'resultado.json')['auxiliary_sha256']
with np.load(source/'auxiliares_Train.npz') as z:q={k:z[k] for k in ['row','train_index','timestamp','true']}
cutoff=int(q['timestamp'][40000:].min());assert (q['timestamp'][:40000]+q['true'][:40000]).max()<cutoff
d,X,names=feature_input('train',read(BEST/'resultado.json'));keep=np.array([not name.startswith('mm_') for name in names]);assert keep.sum()==206
X=X[q['train_index']][:,keep];dd={k:d[k][q['train_index']] for k in ['o_lon','o_lat','d_lon','d_lat','timestamp']};y=q['true']
assert len(y)==50000 and np.array_equal(d['row'][q['train_index']],q['row'])
m=OrderedSourceMemory(source_arrival_before=cutoff);fold=(dd['timestamp']//604800)%5
geometry=np.empty((50000,15),np.float32);hist=np.empty((50000,348),np.float32);proxy=np.empty_like(hist)
parent=np.full((50000,64),-1,np.int32);first=np.zeros((50000,64),np.uint8);last=np.zeros_like(first)
stats=[]
for group in range(6):
 ix=np.flatnonzero((np.arange(50000)<40000)&(fold==group)) if group<5 else np.arange(40000,50000)
 if len(ix)==0:continue
 m.build_tree(group);evaluated=[];discovered=[]
 for a in range(0,len(ix),256):
  selected=ix[a:a+256];queries={k:v[selected] for k,v in dd.items()}
  result=m.query(queries,training=group<5,diagnostics=True)
  g,h,p=source_features(m.data,queries,result);geometry[selected]=g;hist[selected]=h;proxy[selected]=p
  parent[selected]=result[0];first[selected]=result[1];last[selected]=result[2]
  available=result[0]>=0
  if group<5:assert np.all(m.data['parent_fold'][result[0][available]]!=group)
  assert m.available[result[0][available]].all()
  evaluated.extend(result[7]);discovered.extend(result[8])
 stats.append(dict(fold=group,n=len(ix),median_evaluated=float(np.median(evaluated)),median_discovered=float(np.median(discovered))))
 print(json.dumps(dict(stage='crossfit_source_group_complete',**stats[-1],seconds=time.time()-start)),flush=True)
# Rebuild the held tree for an actual raw5 feature order/singleton proof before model fitting.
m.build_tree(5);i=np.arange(40000,40017);queries={k:v[i] for k,v in dd.items()}
res=m.query(queries);g,h,p=source_features(m.data,queries,res)
assert np.array_equal(g,geometry[i]) and np.array_equal(h,hist[i]) and np.array_equal(p,proxy[i])
revq={k:v[::-1] for k,v in queries.items()};rg,rh,rp=source_features(m.data,revq,m.query(revq))
single={k:v[:1] for k,v in queries.items()};sg,sh,sp=source_features(m.data,single,m.query(single))
assert np.array_equal(g,rg[::-1]) and np.array_equal(h,rh[::-1]) and np.array_equal(p,rp[::-1])
assert np.array_equal(g[:1],sg) and np.array_equal(h[:1],sh) and np.array_equal(p[:1],sp)
write(run/'verificacion_fuentes_features.json',dict(passed=True,raw17_cached_summaries_exact=True,reverse_singleton_exact=True,
 source_labels_only_from_arrivedTrain=True,parentweekexcluded_before_finiteK=True,statistics=stats,constructor_source_sha256=sha(OUT/'recuperar_fuentes_ordenadas.py')))
np.savez_compressed(run/'indices_fuentes.npz',row=q['row'],timestamp=q['timestamp'],parent=parent,first=first,last=last)
assert (run/'indices_fuentes.npz').stat().st_size<30*1024**2
m.tree=None;m.tree_ids=None;del m;gc.collect()
params=dict(iterations=1000,depth=8,learning_rate=.05,l2_leaf_reg=5,loss_function='MAE',random_seed=20261124,
 thread_count=2,task_type='GPU',devices='0',gpu_ram_part=.25,border_count=128,allow_writing_files=False,verbose=False)
models={};preds={};metrics={}
for name,value in [('baseline',X),('geometry_CDF_control',np.column_stack([X,geometry,proxy])),('historical_CDF',np.column_stack([X,geometry,hist]))]:
 model=CatBoostRegressor(**params);model.fit(value[:40000],y[:40000].astype(float)/600)
 pred=(model.predict(value[40000:],thread_count=2)*600).clip(60,1785)
 assert np.array_equal(pred[:17],(model.predict(value[40000:40017][::-1],thread_count=2)*600).clip(60,1785)[::-1])
 assert np.array_equal(pred[:1],(model.predict(value[40000:40001],thread_count=2)*600).clip(60,1785))
 models[name]=model;preds[name]=pred;metrics[name]=metric(y[40000:],pred)
 print(json.dumps(dict(stage=name+'_complete',seconds=time.time()-start)),flush=True)
reference=dict(np.load(source/'predicciones_held_Train.npz'))
for key in ['row','timestamp']:assert np.array_equal(reference[key],q[key][40000:])
assert np.array_equal(reference['true'],y[40000:])
comparisons={name:paired(y[40000:],preds['historical_CDF'],other,q['timestamp'][40000:]) for name,other in
 [('vs_baseline',preds['baseline']),('vs_geometry_control',preds['geometry_CDF_control']),('vs_reference',reference['control'])]}
for item in comparisons.values():item['interpretation']='Exploratory repeatedly reused chronological heldTrain, not primaryValidation or confirmation.'
passed=all(comparisons[key]['delta_mae_s']<=-amount and comparisons[key]['ci95_s'][1]<0 for key,amount in
 [('vs_baseline',5),('vs_geometry_control',2),('vs_reference',5)])
np.savez_compressed(run/'predicciones_held_Train.npz',row=q['row'][40000:],timestamp=q['timestamp'][40000:],true=y[40000:],reference=reference['control'],**preds)
saved=[]
if passed:
 for name,model in models.items():
  path=run/(name+'.cbm');model.save_model(str(path));saved.append(dict(name=name,file=path.name,sha256=sha(path)))
result=dict(complete=True,passed=bool(passed),metrics=metrics,comparisons=comparisons,reference=metric(y[40000:],reference['control']),
 sources={name:sha(run/'fuentes'/name) for name in sources},predictions_sha256=sha(run/'predicciones_held_Train.npz'),
 sourceindices_sha256=sha(run/'indices_fuentes.npz'),weights=saved,feature_n=569,original_rows_labels_preserved=True,
 no_model_pretrained_on_heldlabels=True,queryraw5_only=True,sourcefold_exclusion_beforetree=True,
 Validation_access=False,Test_access=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start,
 interpretation='Coverageinstrument led to a fixedtimepredictiongate. No fullprimaryMAE or confirmation claim.',
 next_if_fail='Retire this CDF summary recipe; do not infer lower primaryMAE from geometriccoverage or sweep grids/counts.')
write(run/'resultado.json',result)
assert sum(p.stat().st_size for p in run.rglob('*') if p.is_file())<150*1024**2
print(json.dumps(read(run/'resultado.json')),flush=True)
