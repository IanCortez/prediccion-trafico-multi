"""Actual42millionpoint CPU/RAM/ordered-source proof, Trainqueries only."""
from base_replanteo import *
import gc
from recuperar_subtrayectos import SubtrajectoryMemory
access=deny_test();start=time.time();bank=SubtrajectoryMemory();z=bank.data;tick=time.time();bank.build_tree(0);build_s=time.time()-tick
d=dict(np.load(DATA/'train_x.npz'));eligible=z['eligible'];pool=eligible[((d['timestamp'][eligible]//604800)%5)==0]
ii=np.sort(np.random.default_rng(6321).choice(pool,2048,replace=False));fields=['o_lon','o_lat','d_lon','d_lat','timestamp']
def subset(indices):return {k:d[k][indices] for k in fields}
# Warmcompile on actual parents before timing, no GPU.
warm=bank.query(subset(ii[:8]),training=True)
tick=time.time();parts=[[] for _ in range(7)]
for first in range(0,len(ii),128):
 out=bank.query(subset(ii[first:first+128]),training=True)
 for target,value in zip(parts,out):target.append(value)
out=tuple(np.concatenate(a) for a in parts);query_s=time.time()-tick
parent,i,j,duration,weight,error,full=out;valid=parent>=0;safe=np.maximum(parent,0)
assert np.all(z['parent_fold'][safe[valid]]!=0)
assert np.all(j[valid].astype(int)-i[valid].astype(int)>=4)
assert np.all(j[valid]<z['parent_gps_n'][safe[valid]])
assert np.all(z['timestamp'][safe[valid]]+15*j[valid].astype(np.int64)<1398345209)
# Every source label still15*(j-i); returnedtime differsONLYbypublicODdistance transport.
raw_time=15*(j.astype(np.int32)-i.astype(np.int32))
assert np.all((duration[valid]/raw_time[valid]>=.5-1e-6)&(duration[valid]/raw_time[valid]<=2+1e-6))
firstout=bank.query(subset(ii[:17]),training=True)
reverse=bank.query(subset(ii[:17][::-1]),training=True)
single=bank.query(subset(ii[:1]),training=True)
for a,b,c in zip(firstout,reverse,single):assert np.array_equal(a,b[::-1]) and np.array_equal(a[:1],c)
# Wholeexcludedfold has no treepoint: modifying its storedgeometry cannot changequeries.
bad=np.flatnonzero(z['parent_fold'][bank.parent]==0)
before=tuple(v.copy() for v in firstout);save=z['xy_m'][bad].copy();z['xy_m'][bad]+=100000
after=bank.query(subset(ii[:17]),training=True)
for a,b in zip(before,after):assert np.array_equal(a,b)
z['xy_m'][bad]=save;del save,bad;gc.collect()
counts=valid.sum(1)
result=dict(passed=True,scope='2048Trainquerysourcecoverage/resourceinstrument, not a primaryMAE.',
 source_points=len(bank.parent),tree_points=len(bank.tree_ids),tree_build_s=build_s,
 query_n=len(ii),query_s=query_s,queries_per_second=len(ii)/query_s,
 forecast_fullValidation_retrieval_minutes=query_s/len(ii)*127941/60,
 matching_parent_counts_quantiles=np.quantile(counts,[0,.1,.25,.5,.75,.9,1]),
 no_source_fraction=float(np.mean(counts==0)),fraction_with8parents=float(np.mean(counts>=8)),
 matched_source_ODerror_m_quantiles=np.quantile(error[valid],[.1,.5,.9]) if valid.any() else [],
 source_fulltrip_fraction=float(full[valid].mean()) if valid.any() else None,
 source_order_and15second_labels_verified=True,source_parent_fold_excluded_beforetree=True,
 excluded_fold_geometry_poisoning_invariant=True,actual_query_reverse_singleton_all_sourceoutputs_exact=True,
 no_Validationlabels_or_Test=True,source_sha256=sha(__file__),model_source_sha256=sha(OUT/'recuperar_subtrayectos.py'),
 actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(OUT/'verificacion_memoria_subtrayectos.json',result)
print(json.dumps(read(OUT/'verificacion_memoria_subtrayectos.json')),flush=True)
