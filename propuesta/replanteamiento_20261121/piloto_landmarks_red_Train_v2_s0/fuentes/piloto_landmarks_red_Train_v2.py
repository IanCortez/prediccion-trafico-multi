"""Matched public-topology versus Euclidean references on chronological heldTrain, fixed settings."""
from base_replanteo import *
import shutil,gc
from catboost import CatBoostRegressor
from inferencia import feature_input
from landmarks_red_publica_v2 import PublicLandmarks
access=deny_test();start=time.time();run=OUT/'piloto_landmarks_red_Train_v2_s0';assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
sources=['piloto_landmarks_red_Train_v2.py','landmarks_red_publica_v2.py','base_replanteo.py']
for name in sources:shutil.copy2(OUT/name,run/'fuentes'/name)
plan=dict(scope='Exploratoryreusedchronological40kfit/10kheldTrain, not primaryValidation orconfirmation.',
 hypothesis='Directednetworkdistanceprofiles provideglobaltopologicalODrelations beyondcoordinates andfewexplicitshortestroute summaries.',
 static='32farthestEuclideanreferencejunctions inlargest2014SCC andprimarybbox, selectedwithoutlabels. ExistinglengthDreadonly, no newallpairmatrix.',
 geometry='Fourdirectedendpoint-to/from-reference profiles, marginalizefourpublicGPSprojectioncandidates withsigma30m andaccesserror; log1p(km),132columns.',
 control='Same132columncount, Euclideanreference distances instead ofnetworkdistance, availability1. Directionduplicates areintentional isotropiccontrol; notidenticaleffectivefeatureentropy.',
 baseline='Stored206publicCatBoostreference133.355393, exactsamefit/heldrows andpurgedcutoff.',
 time_model=dict(iterations=1000,depth=8,learning_rate=.05,l2_leaf_reg=5,loss='MAE',target='T/600; forecast*600, originallabelsseconds'),
 gate='>=2secondsbetterthan storedreference AND>=1secondbetterthan matchedEuclidean withupperdayCI<0; no new referencecount/forestdepth sweep.',
 info='Raw5 query only; no heldGPSroute/N/T. All10mm_features excludedfrom206base.',
 resources='OneGPU .25cap, CPU2; <=100MiB pilotartifact; preserveallmetrics/predictions, weights onlyifgatepasses.',
 references=['https://arxiv.org/abs/1309.4973','https://www.sciencedirect.com/science/article/abs/pii/S037843712400726X'],
 literature_limit='Ownstaticgeometryencoding, notfulltime-dependentdistanceoracle orAGEtensor/graphreplica.',
 Validation=False,Test=False)
write(run/'plan.json',plan)
land=PublicLandmarks();comp=land.comp;D=land.D
# Actualgraphdistance triangle inequality, independent offorecasttargets.
edges=np.random.default_rng(811).choice(len(comp['chain_src']),2048,replace=False)
u=comp['chain_src'][edges];v=comp['chain_dst'][edges];L=comp['chain_length'][edges]
left=land.out[u];right=L[:,None]+land.out[v];finite=np.isfinite(left)&np.isfinite(right)
assert np.max(left[finite]-right[finite])<.5
source=OUT/'oraculo_tiempo_calle_train_s0'
with np.load(source/'auxiliares_Train.npz') as z:q={key:z[key] for key in ['row','train_index','timestamp','true']}
d,X,names=feature_input('train',read(BEST/'resultado.json'));keep=np.array([not name.startswith('mm_') for name in names]);assert keep.sum()==206
X=X[q['train_index']][:,keep];dd={key:d[key][q['train_index']] for key in ['o_lon','o_lat','d_lon','d_lat','timestamp']}
meta=read(OUT/'banco_valor_vial_s0/resultado.json');file=OUT/'banco_valor_vial_s0/train_proyecciones.npz'
assert sha(file)==meta['outputs'][file.name]['sha256']
with np.load(file) as z:
 assert np.array_equal(z['row'][q['train_index']],q['row'])
 p={key:z[key][q['train_index']] for key in ['chain','fraction','error_m']}
topo=[];euclid=[]
for first in range(0,len(X),512):
 a,b=land.features({key:value[first:first+512] for key,value in dd.items()},{key:value[first:first+512] for key,value in p.items()})
 topo.append(a);euclid.append(b)
topo=np.concatenate(topo);euclid=np.concatenate(euclid)
# Access-adjusted graphdistance may not be shorter than straightline, allowing canonicalprojectionrounding.
assert np.min(np.expm1(topo[:,:128])*1000-np.expm1(euclid[:,:128])*1000)>-5
ii=np.sort(np.random.default_rng(812).choice(len(X),17,replace=False))
items=[{key:int(dd[key][i]) if key=='timestamp' else float(dd[key][i]) for key in dd} for i in ii]
rd,rp=land.raw_projection(items)
for key in p:assert np.array_equal(rp[key],p[key][ii]),key
ra,rb=land.features(rd,rp);assert np.array_equal(ra,topo[ii]) and np.array_equal(rb,euclid[ii])
rev,revp=land.raw_projection(items[::-1]);a,b=land.features(rev,revp)
single,singlep=land.raw_projection(items[:1]);sa,sb=land.features(single,singlep)
assert np.array_equal(a[::-1],ra) and np.array_equal(b[::-1],rb) and np.array_equal(sa,ra[:1]) and np.array_equal(sb,rb[:1])
write(run/'verificacion_publica.json',dict(passed=True,landmark_nodes=land.landmarks,graph_triangle_proof=True,raw17_cached_features_exact=True,
 reverse_singleton_exact=True,staticmap_only=True,not_an_ODpredictionmetric=True))
truth=q['true'][40000:];reference=dict(np.load(source/'predicciones_held_Train.npz'))
for key in ['row','timestamp']:assert np.array_equal(reference[key],q[key][40000:])
assert np.array_equal(reference['true'],truth) and np.max(q['timestamp'][:40000]+q['true'][:40000])<q['timestamp'][40000:].min()
params=dict(iterations=1000,depth=8,learning_rate=.05,l2_leaf_reg=5,loss_function='MAE',random_seed=20261122,
 thread_count=2,task_type='GPU',devices='0',gpu_ram_part=.25,border_count=128,allow_writing_files=False,verbose=False)
models={};predictions={};metrics={}
for key,extra in [('euclidean',euclid),('directed_network',topo)]:
 value=np.column_stack([X,extra]);m=CatBoostRegressor(**params)
 m.fit(value[:40000],q['true'][:40000].astype(float)/600)
 pred=(m.predict(value[40000:],thread_count=2)*600).clip(60,1785)
 assert np.array_equal(pred[:17],(m.predict(value[40000:40017][::-1],thread_count=2)*600).clip(60,1785)[::-1])
 assert np.array_equal(pred[:1],(m.predict(value[40000:40001],thread_count=2)*600).clip(60,1785))
 models[key]=m;predictions[key]=pred;metrics[key]=metric(truth,pred);del value;gc.collect()
 print(json.dumps(dict(stage='complete_'+key,seconds=time.time()-start)),flush=True)
comparisons=dict(vs_euclidean=paired(truth,predictions['directed_network'],predictions['euclidean'],q['timestamp'][40000:]),
 vs_reference=paired(truth,predictions['directed_network'],reference['control'],q['timestamp'][40000:]))
for value in comparisons.values():value['interpretation']='ExploratoryreusedchronologicalheldTrain, staticgraphfeatures; not primaryValidation orconfirmation.'
passed=all(comparisons[key]['delta_mae_s']<=-delta and comparisons[key]['ci95_s'][1]<0 for key,delta in [('vs_euclidean',1),('vs_reference',2)])
weights=[]
if passed:
 for key,m in models.items():
  path=run/(key+'.cbm');m.save_model(str(path));weights.append(dict(name=key,file=path.name,sha256=sha(path)))
np.savez_compressed(run/'predicciones_held_Train.npz',row=q['row'][40000:],timestamp=q['timestamp'][40000:],true=truth,
 reference=reference['control'],**predictions)
np.savez_compressed(run/'landmarks.npz',junction_id=land.landmarks,xy_m=land.landmark_xy)
result=dict(complete=True,passed=bool(passed),metrics=metrics,reference=metric(truth,reference['control']),comparisons=comparisons,
 features_count=338,public_static_landmark_n=32,raw17_source_features_order_singleton_exact=True,forecast_order_singleton_exact=True,
 no_actual_queryroute_N_T=True,weights=weights,sources={name:sha(run/'fuentes'/name) for name in sources},
 predictions_sha256=sha(run/'predicciones_held_Train.npz'),Validation_access=False,Test_access=False,
 actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(run/'resultado.json',result)
assert sum(p.stat().st_size for p in run.rglob('*') if p.is_file())<100*1024**2
print(json.dumps(read(run/'resultado.json')),flush=True)
