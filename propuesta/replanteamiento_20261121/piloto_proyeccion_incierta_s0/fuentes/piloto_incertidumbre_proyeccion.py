"""Train-only geometric oracle gate: test hypothesis coverage before another GPU model."""
from base_replanteo import *
import shutil
from inferencia import feature_input
from rutas_proyeccion_incierta import build_uncertain_routes
access=deny_test();start=time.time();run=OUT/'piloto_proyeccion_incierta_s0'
assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
for name in ['piloto_incertidumbre_proyeccion.py','rutas_proyeccion_incierta.py','consulta_historicos_cruzados.py','base_replanteo.py']:
 shutil.copy2(OUT/name,run/'fuentes'/name)
protocol=dict(seed=20261122,n=20000,scope='Train-only oracle geometry coverage, not OD TTE performance or an error lower bound.',
 hypotheses='Protected two default routes plus two cost types for three additional distinct endpoint-projection pairs selected by squared GPS projection residual, independent of labels.',
 information='Raw five OD/departure fields; whole UTC-week historical exclusion for Train; actual geometry only evaluation target.',
 gate='At least2percent and5meters mean seven-point geometry-oracle reduction vs protected two routes plus straight-line prior.',
 Validation_labels_accessed=False,Test='sealed')
write(run/'protocolo.json',protocol)
rec=read(BEST/'resultado.json');tr,X,names=feature_input('train',rec)
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index'];gt=dict(np.load(DATA/'geometria_train.npz'))
pos=np.searchsorted(gt['row'],tr['row'][eligible]);safe=np.minimum(pos,len(gt['row'])-1)
good=(pos<len(gt['row']))&(gt['row'][safe]==tr['row'][eligible])&(gt['X'][safe,-1]<=500)&np.isfinite(gt['X'][safe,1:15]).all(1)
sample=np.sort(np.random.default_rng(20261122).choice(eligible[good],20000,replace=False))
gp=np.searchsorted(gt['row'],tr['row'][sample]);geometry=gt['X'][gp,1:15];actual_length=gt['X'][gp,0]
keys=['o_lon','o_lat','d_lon','d_lat','timestamp']
items=[{k:int(tr[k][i]) if k=='timestamp' else float(tr[k][i]) for k in keys} for i in sample]
B,pairs=build_uncertain_routes(items,training=True)
# The original two protected route hypotheses must be present before considering expansion.
check=[]
for kind,tag in enumerate(['length','free']):
 original=[names.index('projected_'+tag+'_shape_'+str(i)) for i in range(14)]+[names.index('projected_'+tag+'_'+v) for v in ['polyline_km','circuity_log','turn_abs_pi','turn_signed_pi','sharp_turn_count','max_turn_pi','available','freeflow_600s','o_projection_km','d_projection_km']]
 original+=[names.index('mm_'+tag+'_'+v) for v in ['gps_time_600s','state_support','link_support','congestion_log','pace_log']]
 bank_cols=list(range(24))+list(range(26,31));valid=B[:,kind,20]>0
 available=X[sample,names.index('projected_'+tag+'_available')]>0
 assert np.array_equal(valid,available)
 error=float(np.abs(B[valid,kind][:,bank_cols]-X[sample[valid]][:,original]).max())
 assert error<1e-5;check.append(error)
small,_=build_uncertain_routes(items[:5],training=True)
reverse,_=build_uncertain_routes(items[:5][::-1],training=True)
assert np.array_equal(small,B[:5]) and np.array_equal(small,reverse[::-1])
rejected=False
try:build_uncertain_routes([dict(items[0],actual_path=[])],training=True)
except AssertionError:rejected=True
assert rejected
shape=B[:,:,:14].reshape(len(B),8,7,2);truth=geometry.reshape(len(B),1,7,2)
distance=np.sqrt(np.square(shape-truth).sum(-1)).mean(-1)*1000
distance[B[:,:,20]==0]=np.inf
straight=np.sqrt(np.square(truth[:,0]).sum(-1)).mean(-1)*1000
baseline=np.minimum(distance[:,:2].min(1),straight);expanded=np.minimum(distance.min(1),straight)
assert np.all(expanded<=baseline+1e-5)
mean_reduction=float(np.mean(baseline-expanded));relative=mean_reduction/max(float(np.mean(baseline)),1e-9)
length_error=np.abs(B[:,:,14]*1000-actual_length[:,None]);length_error[B[:,:,20]==0]=np.inf
default_length=np.minimum(length_error[:,:2].min(1),np.abs(tr['od_m'][sample]-actual_length))
expanded_length=np.minimum(length_error.min(1),np.abs(tr['od_m'][sample]-actual_length))
passed=mean_reduction>=5 and relative>=.02
np.savez_compressed(run/'predicciones_oraculo_Train.npz',row=tr['row'][sample],timestamp=tr['timestamp'][sample],
 geometry_baseline_m=baseline,geometry_expanded_m=expanded,length_baseline_m=default_length,length_expanded_m=expanded_length,
 candidate_geometry_error_m=distance,candidate_available=B[:,:,20].astype(bool),projection_pairs=pairs)
r=dict(passed=bool(passed),n=len(sample),mean_geometry_baseline_m=float(baseline.mean()),mean_geometry_expanded_m=float(expanded.mean()),
 mean_geometry_reduction_m=mean_reduction,relative_geometry_reduction=relative,fraction_improving_more10m=float(np.mean(baseline-expanded>10)),
 mean_length_baseline_m=float(default_length.mean()),mean_length_expanded_m=float(expanded_length.mean()),
 original_two_route_feature_recovery_max_error=check,reverse_and_batch_subset_exact=True,actual_query_path_rejected=True,
 source_sha256={p.name:sha(p) for p in (run/'fuentes').glob('*.py')},predictions_sha256=sha(run/'predicciones_oraculo_Train.npz'),
 primary_filters_unchanged=True,Validation_access=False,Test_access=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start,
 conclusion='Gate supports or rejects expanding public endpoint-projection hypotheses. Oracle uses Train geometry and is not a deployable selector, primary MAE result, or universal bound.')
write(run/'resultado.json',r);print(json.dumps({k:r[k] for k in ['passed','n','mean_geometry_baseline_m','mean_geometry_expanded_m','mean_geometry_reduction_m','relative_geometry_reduction','fraction_improving_more10m','seconds']}),flush=True)
