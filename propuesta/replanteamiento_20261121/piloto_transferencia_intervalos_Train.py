"""Boundary-aware, density-weighted subtrip augmentation; chronological Train-only gate."""
from base_replanteo import *
import shutil,gc
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,log_loss
from threadpoolctl import threadpool_limits
from inferencia import feature_input
from consulta_od import od_records,build_od_features
from consulta_historicos_cruzados import prepare,history_features,STATIC_NAMES

access=deny_test();start=time.time();run=OUT/'piloto_transferencia_intervalos_Train_s0'
assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
sources=['piloto_transferencia_intervalos_Train.py','consulta_historicos_cruzados.py','base_replanteo.py']
for name in sources:shutil.copy2(OUT/name,run/'fuentes'/name)
plan=dict(scope='Reused chronological 40000fit/10000heldTrain; no primaryValidation or independent confirmation.',
 hypothesis='Learn directly from observed interval OD/time pairs with endpoint-boundary indicators and bounded covariate-density weighting, rather than retrieval or warm-start neural auxiliary heads.',
 distinction='Previous auxiliary experiment warmed a separate neural head and updated a pretrained encoder with fixed sourceweight. Here use fresh shared tree partitions, explicit boundary covariates and cross-fitted density-ratio weights.',
 parent_selection='Only the exact40000 original fitparents; their arrival is before the first held departure. Heldparent geometry never selected.',
 interval_sampling='Three seeded proper intervals per fitparent, no enumeration. Interval labels15*(j-i); timestamp=parenttimestamp+15*i.',
 auxiliary_filters='Original public bbox/OD>=200m/ODspeed<=120kmh/time60..1785 applied only to sourceexamples. All primary fit/heldrows and labels stay unchanged.',
 source_covariates='206public features excluding every mm_ profile; plus whether sourceorigin/destination is its parentendpoint. Bothflags1 for complete querytrips, never queryN/T.',
 density='Five OOF parent-group splits. Balanced logisticdomain classifier on206publicX, per-fold scaler; clip posterior odds .05..10 and normalize source mean1. Source totalweight=.2*full weight.',
 identification_limit='Density weighting assumes conditional transfer after boundary adjustment; this is a hypothesis, not a guaranteed correction of label shift.',
 regressors=dict(iterations=1000,depth=8,learning_rate=.05,l2_leaf_reg=5,loss='MAE',target='Originalseconds/600 then inverse600'),
 control='Same sourceX/flags/weights but auxiliarytargets are public baselinepredictions. Distinguishes actual source-time labels from regularization by extra covariates.',
 gate='>=5secondsbetter than no-augmentationbaseline ANDstoredreference; >=2secondsbetter than pseudo-labelcontrol, dayCIupper<0 in all three. No sourceweight/count/depth sweep.',
 resources='OneGPU .25cap,CPU2. Sourcefeatures ephemeral; persist only provenance, metrics, predictions and sources; pilotweights onlyifgatepasses.',
 references=['https://arxiv.org/html/2407.05895v2'],
 literature_limit='ProbETA motivates trip-subsampling but takes queriedpath/context; this OD adaptation is not a replica or a comparable claimedscore.',
 query=['o_lon','o_lat','d_lon','d_lat','timestamp'],Validation=False,Test=False)
write(run/'plan.json',plan)
source=OUT/'oraculo_tiempo_calle_train_s0';meta=read(source/'resultado.json')
assert sha(source/'auxiliares_Train.npz')==meta['auxiliary_sha256']
with np.load(source/'auxiliares_Train.npz') as z:q={k:z[k] for k in ['row','train_index','timestamp','true']}
d,X,names=feature_input('train',read(BEST/'resultado.json'))
keep=np.array([not n.startswith('mm_') for n in names]);assert keep.sum()==206
X=X[q['train_index']][:,keep];public_names=[n for n,k in zip(names,keep) if k]
y=q['true'];cutoff=int(q['timestamp'][40000:].min());fitparent=q['train_index'][:40000]
assert len(np.unique(fitparent))==40000 and (q['timestamp'][:40000]+y[:40000]).max()<cutoff
mem=OUT/'memoria_puntos_Train_s0/puntos_Train.npz'
assert sha(mem)==read(mem.parent/'resultado.json')['outputs_sha256']
with np.load(mem) as z:
 offsets=z['offsets'];counts=z['parent_gps_n'];ts=z['timestamp']
 assert np.array_equal(z['row'][q['train_index']],q['row'])
 assert np.array_equal(z['time_s'][q['train_index']],y)
 # Existing immutable Train-only pointbank. Choose parent IDs before selecting geometry.
 points=z['xy_m'];assert np.all(ts[fitparent]+15*(counts[fitparent]-1)<cutoff)
 sx=111195*np.cos(np.deg2rad(41.16));sy=111195.
 rg=np.random.default_rng(20261123);rows=[];auxy=[];parents=[];first=[];last=[];flags=[]
 bbox=read(DATA/'manifiesto.json')['bbox'];lat0,lat1,lon0,lon1=bbox
 excluded={}
 for parent in fitparent:
  n=int(counts[parent]);pairs=set()
  if n<6:excluded['parent_too_short']=excluded.get('parent_too_short',0)+1;continue
  for attempt in range(3):
   span=int(rg.integers(4,n-1));i=int(rg.integers(0,n-span));j=i+span
   if (i,j) in pairs:continue
   pairs.add((i,j))
   a=points[int(offsets[parent])+i].astype(float);b=points[int(offsets[parent])+j].astype(float)
   olon=a[0]/sx-8.62;olat=a[1]/sy+41.16;dlon=b[0]/sx-8.62;dlat=b[1]/sy+41.16
   assert 4<=j-i<=119 and not(i==0 and j==n-1)
   if not(lon0<=olon<=lon1 and lon0<=dlon<=lon1 and lat0<=olat<=lat1 and lat0<=dlat<=lat1):continue
   dy=np.deg2rad(dlat-olat);dx=np.deg2rad(dlon-olon)
   h=np.sin(dy/2)**2+np.cos(np.deg2rad(olat))*np.cos(np.deg2rad(dlat))*np.sin(dx/2)**2
   dist=6371000*2*np.arcsin(min(1,np.sqrt(h)));t=15*(j-i)
   if dist<200 or dist/t*3.6>120:continue
   stamp=int(ts[parent])+15*i
   assert stamp+t<=int(ts[parent])+15*(n-1)<cutoff
   rows.append((olon,olat,dlon,dlat,stamp));auxy.append(t);parents.append(int(parent));first.append(i);last.append(j);flags.append((i==0,j==n-1))
 del points
Aquery=np.asarray(rows,np.float64);ay=np.asarray(auxy,np.float64);aflags=np.asarray(flags,np.float32)
assert len(ay)>10000 and len(ay)<=120000 and np.isin(parents,fitparent).all()
assert not np.isin(parents,q['train_index'][40000:]).any()
np.savez_compressed(run/'procedencia_intervalos.npz',query=Aquery,time_s=ay.astype(np.int16),parent_train_index=np.asarray(parents,np.int32),first=np.asarray(first,np.uint8),last=np.asarray(last,np.uint8),boundary_flags=aflags)
rec=read(BEST/'resultado.json');cfg=dict(rec['config']);cfg.update(projected_routes=False,mapmatched=False,learned_route=False,road_attributes=False)
def public_features(items):
 data=od_records(items);b,bn=build_od_features(data,cfg);prepared=prepare(items)
 movement,static=history_features(prepared,5)
 # Movement values are NEVER appended or used. Static routes must be independent of this profile.
 return np.column_stack([b,static]).astype(np.float32),bn+STATIC_NAMES
def items_from_array(a):
 return [{k:int(v) if k=='timestamp' else float(v) for k,v in zip(plan['query'],row)} for row in a]
# Instrument proof before any regression fit: exact same five-field public representation.
ii=np.arange(17);items=[{k:int(d[k][q['train_index'][i]]) if k=='timestamp' else float(d[k][q['train_index'][i]]) for k in plan['query']} for i in ii]
probe,pnames=public_features(items);assert pnames==public_names
error=float(np.max(np.abs(probe-X[ii])));assert np.allclose(probe,X[ii],atol=2e-5,rtol=2e-5),error
reverse,_=public_features(items[::-1]);single,_=public_features(items[:1])
assert np.array_equal(probe,reverse[::-1]) and np.array_equal(probe[:1],single)
prepared=prepare(items);_,s0=history_features(prepared,0);_,s5=history_features(prepared,5);assert np.array_equal(s0,s5)
write(run/'verificacion_publica.json',dict(passed=True,raw17_feature_maxerror=error,reverse_singleton_exact=True,
 static_profile_independent=True,no_movement_values_in_features=True,only_fitparent_intervals=True,
 auxiliary_n=len(ay),selected_parents=40000,source_coordinates='ImmutableTrainF32meters inverted to lonlat; millimeter-scale coordinate approximation.',
 cutoff=cutoff,no_held_route_N_T_as_input=True))
parts=[]
for a in range(0,len(Aquery),4096):
 value,nms=public_features(items_from_array(Aquery[a:a+4096]));assert nms==public_names;parts.append(value)
 if len(parts)%8==0:print(json.dumps(dict(stage='build_public_interval_features',n=min(a+4096,len(Aquery)),seconds=time.time()-start)),flush=True)
A=np.concatenate(parts);del parts;gc.collect()
assert A.shape==(len(ay),206) and np.isfinite(A).all()
# Learn density ratios using ONLY fitparents, grouped out-of-fold.
group={int(p):int(g) for p,g in zip(fitparent,np.random.default_rng(851).permutation(len(fitparent))%5)}
groups=np.r_[np.array([group[int(p)] for p in fitparent]),np.array([group[int(p)] for p in parents])]
domain=np.r_[np.ones(40000),np.zeros(len(A))];Z=np.row_stack([X[:40000],A]);posterior=np.zeros(len(Z))
with threadpool_limits(limits=2):
 for fold in range(5):
  fit=groups!=fold;held=~fit;scaler=StandardScaler().fit(Z[fit]);clf=LogisticRegression(C=1.,max_iter=500,class_weight='balanced',random_state=851)
  clf.fit(scaler.transform(Z[fit]),domain[fit]);assert clf.n_iter_[0]<500
  posterior[held]=clf.predict_proba(scaler.transform(Z[held]))[:,1]
assert np.isfinite(posterior).all()
odds=np.clip(posterior[40000:]/np.clip(1-posterior[40000:],1e-6,None),.05,10)
sourceweight=odds/odds.mean()*.2*40000/len(A)
assert abs(sourceweight.sum()-8000)<1e-6
write(run/'diagnostico_transferencia.json',dict(domain_auc=float(roc_auc_score(domain,posterior)),domain_crossentropy=float(log_loss(domain,posterior)),
 ratio_quantiles=np.quantile(odds,[0,.1,.5,.9,1]),effective_source_n=float(sourceweight.sum()**2/(sourceweight**2).sum()),
 weights_total=float(sourceweight.sum()),source_time_mean_s=float(ay.mean()),full_time_mean_s=float(y[:40000].mean()),
 limit='OOFdomain discrimination does not prove conditional label-transferability; sourceboundary indicators may not remove all behavioral differences.'))
del Z;gc.collect()
fitX=np.column_stack([X[:40000],np.ones((40000,2),np.float32)]);heldX=np.column_stack([X[40000:],np.ones((10000,2),np.float32)])
auxX=np.column_stack([A,aflags]);del A;gc.collect()
params=dict(iterations=1000,depth=8,learning_rate=.05,l2_leaf_reg=5,loss_function='MAE',random_seed=20261123,
 thread_count=2,task_type='GPU',devices='0',gpu_ram_part=.25,border_count=128,allow_writing_files=False,verbose=False)
models={};preds={};metrics={}
baseline=CatBoostRegressor(**params);baseline.fit(fitX,y[:40000].astype(float)/600)
preds['baseline']=(baseline.predict(heldX,thread_count=2)*600).clip(60,1785);models['baseline']=baseline;metrics['baseline']=metric(y[40000:],preds['baseline'])
pseudo=(baseline.predict(np.column_stack([auxX[:,:206],np.ones((len(auxX),2),np.float32)]),thread_count=2)*600).clip(60,1785)
print(json.dumps(dict(stage='baseline_complete',seconds=time.time()-start)),flush=True)
combined=np.row_stack([fitX,auxX]);weights=np.r_[np.ones(40000),sourceweight]
for name,target in [('pseudo_label_control',pseudo),('observed_intervals',ay)]:
 model=CatBoostRegressor(**params);model.fit(combined,np.r_[y[:40000],target]/600,sample_weight=weights)
 preds[name]=(model.predict(heldX,thread_count=2)*600).clip(60,1785);models[name]=model;metrics[name]=metric(y[40000:],preds[name])
 assert np.array_equal(preds[name][:17],(model.predict(heldX[:17][::-1],thread_count=2)*600).clip(60,1785)[::-1])
 assert np.array_equal(preds[name][:1],(model.predict(heldX[:1],thread_count=2)*600).clip(60,1785))
 print(json.dumps(dict(stage=name+'_complete',seconds=time.time()-start)),flush=True)
reference=dict(np.load(source/'predicciones_held_Train.npz'))
for key in ['row','timestamp']:assert np.array_equal(reference[key],q[key][40000:])
assert np.array_equal(reference['true'],y[40000:])
comparisons={name:paired(y[40000:],preds['observed_intervals'],other,q['timestamp'][40000:])
 for name,other in [('vs_baseline',preds['baseline']),('vs_pseudo_label_control',preds['pseudo_label_control']),('vs_reference',reference['control'])]}
for v in comparisons.values():v['interpretation']='Exploratory repeatedly reused chronological heldTrain; not primaryValidation or confirmation.'
passed=all(comparisons[key]['delta_mae_s']<=-threshold and comparisons[key]['ci95_s'][1]<0 for key,threshold in [('vs_baseline',5),('vs_reference',5),('vs_pseudo_label_control',2)])
np.savez_compressed(run/'predicciones_held_Train.npz',row=q['row'][40000:],timestamp=q['timestamp'][40000:],true=y[40000:],reference=reference['control'],**preds)
weights_saved=[]
if passed:
 for name,model in models.items():
  path=run/(name+'.cbm');model.save_model(str(path));weights_saved.append(dict(name=name,file=path.name,sha256=sha(path)))
result=dict(complete=True,passed=bool(passed),metrics=metrics,reference=metric(y[40000:],reference['control']),comparisons=comparisons,
 auxiliary_n=len(ay),full_fit_n=40000,held_n=10000,all_original_held_rows_labels_preserved=True,
 no_held_GPS_N_T_in_features=True,domain_weighting_only_fitparents=True,boundary_flags_query_constant_full=True,
 predictions_sha256=sha(run/'predicciones_held_Train.npz'),provenance_sha256=sha(run/'procedencia_intervalos.npz'),
 sources={name:sha(run/'fuentes'/name) for name in sources},weights=weights_saved,Validation_access=False,Test_access=False,
 actual_data_files_accessed=sorted(access),seconds=time.time()-start,
 next_if_fail='Retire this boundary/density augmentation recipe. Do not sweep sourceweights/counts/treeparameters.')
write(run/'resultado.json',result)
assert sum(p.stat().st_size for p in run.rglob('*') if p.is_file())<150*1024**2
print(json.dumps(read(run/'resultado.json')),flush=True)
