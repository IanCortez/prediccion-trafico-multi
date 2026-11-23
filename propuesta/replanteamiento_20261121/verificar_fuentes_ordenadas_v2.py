"""Independent interval oracle, pruning equivalence, raw5 and fold-exclusion proofs."""
from base_replanteo import *
import shutil
from recuperar_fuentes_ordenadas import *
from datos_progreso_recurrente import lisbon_transitions
access=deny_test();start=time.time();run=OUT/'verificacion_fuentes_ordenadas_v2_s0'
assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
for name in ['verificar_fuentes_ordenadas_v2.py','recuperar_fuentes_ordenadas.py','base_replanteo.py']:
 shutil.copy2(OUT/name,run/'fuentes'/name)
set_num_threads(2);rg=np.random.default_rng(852)
sizes=rg.integers(8,121,200);sizes[:2]=96;offset=np.r_[0,np.cumsum(sizes)]
parts=[]
for n in sizes:
 base=rg.normal(size=2)*1800
 steps=rg.normal(size=(n,2))*12+np.array([7.,2.])
 parts.append((base+np.cumsum(steps,axis=0)).astype(np.float32))
parts[1]=parts[0].copy();xy=np.concatenate(parts)
indexable=rg.random(len(xy))>.15;indexable[offset[:-1]]=True;indexable[offset[1:]-1]=True
indexable[offset[1]:offset[2]]=indexable[offset[0]:offset[1]]
parent=np.repeat(np.arange(len(sizes),dtype=np.int32),sizes)
ts=rg.integers(1373000000,1396000000,len(sizes),dtype=np.int64);ts[1]=ts[0]
fold=(ts//604800)%5;starts,ends=lisbon_transitions()
boxes=parent_boxes(xy,offset,indexable)
qxy=np.array([np.r_[parts[p][0],parts[p][-1]]+rg.normal(size=4)*5 for p in [0,3,4,10,20]])
qts=np.full(len(qxy),1398000000,np.int64);qfold=np.full(len(qxy),5)
ids=np.flatnonzero(indexable);tree=cKDTree(xy[ids].astype(float));_,oi=tree.query(qxy[:,:2],k=256);_,di=tree.query(qxy[:,2:],k=256);oi=ids[oi];di=ids[di]
args=(oi,di,parent,offset,sizes,ts,fold,xy,indexable,boxes,qxy,qts,qfold,starts,ends)
a=ordered_candidates(*args,True);b=ordered_candidates(*args,False)
for k in list(range(7))+[9]:assert np.array_equal(a[k],b[k]),('pruning',k)
# Completely independent exhaustive feasiblepair search and prior ranking, including indexablemask.
for q in range(len(qxy)):
 expected=[]
 for p in np.unique(np.r_[parent[oi[q]],parent[di[q]]]):
  lo,hi=offset[p:p+2];points=xy[lo:hi];mask=indexable[lo:hi];n=len(points)
  d0=((points.astype(float)-qxy[q,:2])**2).sum(1);d1=((points.astype(float)-qxy[q,2:])**2).sum(1)
  best=np.inf;bi=-1;bj=-1;blen=0.
  for j in range(4,n):
   if not mask[j]:continue
   ii=np.arange(max(0,j-119),j-3);delta=points[j]-points[ii];length=np.sqrt((delta*delta).sum(1))
   valid=mask[ii]&(length>=200)&(length/(15*(j-ii))*3.6<=120)
   for i,L in zip(ii[valid],length[valid]):
    value=d0[i]+d1[j]
    if value<best-1e-10 or(abs(value-best)<=1e-10 and(i<bi or(i==bi and j>bj))):best=value;bi=int(i);bj=j;blen=float(L)
  if bi<0:continue
  sh,sw=clock_vars(int(ts[p])+15*bi,starts,ends);qh,qw=clock_vars(int(qts[q]),starts,ends);delta=(sh-qh+12)%24-12
  w=-.5*best/2500-.5*(delta/4)**2+(0. if sw==qw else np.log(.35))
  expected.append((w,int(p),bi,bj))
 expected.sort(key=lambda v:(-v[0],v[1]));expected=expected[:64]
 actual_n=int((a[0][q]>=0).sum());assert actual_n==len(expected)
 for j,(w,p,i,k) in enumerate(expected):
  assert int(a[0][q,j])==p and int(a[1][q,j])==i and int(a[2][q,j])==k
  assert abs(a[4][q,j]-w)<1e-10
write(run/'oraculo_sintetico.json',dict(passed=True,queries=len(qxy),sourceparents=200,exhaustivepairs_and_weight_ranking=True,
 original_indexable_mask_preserved=True,duplicate_parent_ties_canonical=True,pruned_unpruned_top64_exact=True))
print(json.dumps(dict(stage='independent_synthetic_oracle_passed',seconds=time.time()-start)),flush=True)
# Real heldTrain raw5 sources: sourcearrival cutoff is applied BEFORE candidate trees.
tr=dict(np.load(DATA/'train_x.npz'))
with np.load(OUT/'oraculo_tiempo_calle_train_s0/auxiliares_Train.npz') as z:q={k:z[k] for k in ['row','train_index','timestamp']}
cutoff=int(q['timestamp'][40000:].min());ii=np.arange(40000,40065)
d={k:tr[k][q['train_index'][ii]] for k in ['o_lon','o_lat','d_lon','d_lat','timestamp']}
m=OrderedSourceMemory(source_arrival_before=cutoff);m.build_tree(5)
tick=time.time();a=m.query(d,diagnostics=True,prune=True);pruned_seconds=time.time()-tick
tick=time.time();b=m.query({k:v[:17] for k,v in d.items()},diagnostics=True,prune=False);unpruned17_seconds=time.time()-tick
for k in list(range(7))+[9]:assert np.array_equal(a[k][:17],b[k]),('realpruning',k)
rev=m.query({k:v[:17][::-1] for k,v in d.items()},diagnostics=True)
single=m.query({k:v[:1] for k,v in d.items()},diagnostics=True)
for k in range(10):
 assert np.array_equal(a[k][:17],rev[k][::-1]),('reverse',k)
 assert np.array_equal(a[k][:1],single[k]),('singleton',k)
available=a[0]>=0
assert np.all(m.available[a[0][available]])
for row in range(len(d['timestamp'])):
 for j in np.flatnonzero(available[row]):
  p=int(a[0][row,j]);i=int(a[1][row,j]);k=int(a[2][row,j]);off=m.data['offsets'][p]
  assert 4<=k-i<=119 and m.indexable[off+i] and m.indexable[off+k]
np.savez_compressed(run/'fuentes_65_Train.npz',row=q['row'][ii],parent=a[0],first=a[1],last=a[2],scaled_source_time_s=a[3],logweight=a[4],endpoint_error_m=a[5],
 full_source=a[6],evaluated_parents=a[7],discovered_parents=a[8],global_minerror_m=a[9])
print(json.dumps(dict(stage='real_raw65_pruning_order_singleton_passed',seconds=time.time()-start)),flush=True)
# One real whole-week excluded tree, built with exclusion before finite nearestpoint search.
fold_id=int((d['timestamp'][0]//604800)%5);which=np.flatnonzero((d['timestamp']//604800)%5==fold_id)[:17]
m.build_tree(fold_id);c=m.query({k:v[which] for k,v in d.items()},training=True)
have=c[0]>=0;assert np.all(m.data['parent_fold'][c[0][have]]!=fold_id)
result=dict(passed=True,independent_exhaustive_synthetic_oracle=True,sourceendpoint_bbox_preserved=True,
 real_raw65_top64_sources=True,real17_pruned_unpruned_sources_weights_and_global_minerror_exact=True,
 real65_reverse17_singleton_all10outputs_exact=True,sourcearrival_before_tree=True,whole_parent_UTCweek_exclusion_before_tree=True,
 source_interval_span4to119=True,query=['o_lon','o_lat','d_lon','d_lat','timestamp'],no_query_route_N_T=True,
 pruned65_seconds=pruned_seconds,unpruned17_seconds=unpruned17_seconds,
 evaluated_parents_median=float(np.median(a[7])),discovered_parents_median=float(np.median(a[8])),
 nearest_source_within100m_fraction=float((a[9]<=100).mean()),new_MAE_claim=False,
 sources={p.name:sha(p) for p in (run/'fuentes').glob('*.py')},outputs_sha256=sha(run/'fuentes_65_Train.npz'),
 Validation_access=False,Test_access=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(run/'resultado.json',result);print(json.dumps(read(run/'resultado.json')),flush=True)
