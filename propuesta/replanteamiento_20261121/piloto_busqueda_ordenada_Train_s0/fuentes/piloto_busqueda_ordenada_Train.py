"""Train-only geometry instrument: parent discovery plus exact ordered source search."""
from base_replanteo import *
import gc,shutil
from scipy.spatial import cKDTree
from numba import set_num_threads
from recuperar_subtrayectos import metric_xy
from busqueda_ordenada_fuentes import ordered_parent_support,brute_parent_support

access=deny_test();start=time.time();set_num_threads(2)
run=OUT/'piloto_busqueda_ordenada_Train_s0';assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
sources=['piloto_busqueda_ordenada_Train.py','busqueda_ordenada_fuentes.py','base_replanteo.py']
for name in sources:shutil.copy2(OUT/name,run/'fuentes'/name)
write(run/'plan.json',dict(scope='Outcome-free geometrycoverage on512reusedheldTrainqueries, not MAE.',
 trigger='Extra22.16Mpoints worsened matchedK support:40queries lost vs15gained, demonstrating candidatecompetition in this finite procedure.',
 algorithm='Use same2048pointneighbors perendpoint only to discover unionofhistoricalparents. Within each discoveredparent solve the exact minimum endpointRSS over ordered60..1785s intervals with valid sourceOD/speed. Sliding-window minimum with exact constrained fallback.',
 original='Independent endpoint-topK intersection can discard a source even when another point in that sameparent provides a good ordered match.',
 data='Original pointbank only, sourcearrivalsbefore heldTrain cutoff, no additionalGPSbank or newdata used inthis algorithmtest.',
 gate='>=5percentagepoints more queries with sourcewithin100m RSS, no perquery increase in minimumRSS or loss of existing support. No K/radius tuning.',
 proof='Independent brute-force versus monotone-window implementation including longparents and invalid-minimum fallback.',
 limits='Geometriccoverage does not demonstrate MAEgain. A positive instrument would require frozenTrainconstruction/retraining before any primaryevaluation.',
 Test=False,Validation=False))
# Independent synthetic exactness, including longparents and GPS reversals/repeats.
rg=np.random.default_rng(473)
sizes=np.array([5,12,40,120,300]);offset=np.r_[0,np.cumsum(sizes)]
xy=np.concatenate([np.cumsum(rg.normal(size=(n,2))*70,axis=0) for n in sizes]).astype(np.float32)
xy[20:23]=xy[20];parent=np.repeat(np.arange(len(sizes),dtype=np.int32),sizes)
qxy=rg.normal(size=(7,4))*180
oi=rg.integers(0,len(xy),size=(7,8));di=rg.integers(0,len(xy),size=(7,8))
c,e=ordered_parent_support(oi,di,parent,offset,xy,qxy);bc,be=brute_parent_support(oi,di,parent,offset,xy,qxy)
assert np.array_equal(c,bc) and np.allclose(e,be,rtol=1e-10,atol=1e-8)
tr=dict(np.load(DATA/'train_x.npz'))
with np.load(OUT/'oraculo_tiempo_calle_train_s0/auxiliares_Train.npz') as z:q={k:z[k] for k in ['row','train_index','timestamp']}
qi=40000+np.sort(np.random.default_rng(20261124).choice(10000,512,replace=False));cutoff=int(q['timestamp'][40000:].min())
query={k:tr[k][q['train_index'][qi]] for k in ['o_lon','o_lat','d_lon','d_lat','timestamp']};qxy=metric_xy(query)
source=OUT/'memoria_puntos_Train_s0/puntos_Train.npz';assert sha(source)==read(source.parent/'resultado.json')['outputs_sha256']
z=dict(np.load(source));parent=np.repeat(np.arange(len(z['parent_gps_n']),dtype=np.int32),z['parent_gps_n'])
inside=np.unpackbits(z['indexable_packed'],count=len(parent)).astype(bool)
eligible=(z['parent_gps_n']>=5)&(z['timestamp']+z['time_s']<cutoff)
ids=np.flatnonzero(inside&eligible[parent]);tree=cKDTree(z['xy_m'][ids].astype(np.float64))
od,oi=tree.query(qxy[:,:2],k=2048,workers=2);dd,di=tree.query(qxy[:,2:],k=2048,workers=2)
oi=ids[oi];di=ids[di]
with np.load(OUT/'cobertura_fuentes_ampliadas_Train_s0/cobertura_consultas_Train.npz') as old:
 assert np.array_equal(old['row'],q['row'][qi]);oldcount=old['old_source_count'];olderror=old['old_minerror_m']
# Realdata independent brute-force proof on a smaller candidatebudget; scoring budget remains2048.
c,e=ordered_parent_support(oi[:4,:8],di[:4,:8],parent,z['offsets'],z['xy_m'],qxy[:4])
bc,be=brute_parent_support(oi[:4,:8],di[:4,:8],parent,z['offsets'],z['xy_m'],qxy[:4])
assert np.array_equal(c,bc) and np.allclose(e,be,rtol=1e-10,atol=1e-8)
count,error=ordered_parent_support(oi,di,parent,z['offsets'],z['xy_m'],qxy)
# The original per-parent point-pair set is a strict subset of the new feasible search space.
assert np.all(count>=oldcount)
assert np.all(error<=olderror+1e-7)
reverse_count,reverse_error=ordered_parent_support(oi[:17][::-1],di[:17][::-1],parent,z['offsets'],z['xy_m'],qxy[:17][::-1])
single_count,single_error=ordered_parent_support(oi[:1],di[:1],parent,z['offsets'],z['xy_m'],qxy[:1])
assert np.array_equal(count[:17],reverse_count[::-1]) and np.array_equal(error[:17],reverse_error[::-1])
assert np.array_equal(count[:1],single_count) and np.array_equal(error[:1],single_error)
before=dict(zero_source_fraction=float((oldcount==0).mean()),source_within100m_fraction=float((olderror<=100).mean()))
after=dict(zero_source_fraction=float((count==0).mean()),source_within100m_fraction=float((error<=100).mean()),median_source_count=float(np.median(count)),
 finite_minerror_quantiles=np.quantile(error[np.isfinite(error)],[.1,.5,.9]))
gain=float((error<=100).mean()-(olderror<=100).mean());passed=gain>=.05
np.savez_compressed(run/'cobertura_consultas_Train.npz',row=q['row'][qi],timestamp=query['timestamp'],old_source_count=oldcount,new_source_count=count,old_minerror_m=olderror,new_minerror_m=error)
result=dict(complete=True,passed=bool(passed),before=before,after=after,within100m_gain_percentage_points=100*gain,
 minimumRSS_never_increased=True,original_support_never_lost=True,synthetic_and_real_bruteforce_exact=True,
 reversed_singleton_exact=True,sourceparents_selected_before_tree=True,held_query_time_route_not_used=True,
 additional_GPS_bytes_persisted=0,Test_access=False,Validation_access=False,
 sources={name:sha(run/'fuentes'/name) for name in sources},coverage_sha256=sha(run/'cobertura_consultas_Train.npz'),
 actual_data_files_accessed=sorted(access),seconds=time.time()-start,
 next='Positivegeometrygate requires separate fullTraincrossfit sourceconstructor and independentprototype before any primaryMAEclaim; no threshold/Ksweep.')
write(run/'resultado.json',result);print(json.dumps(read(run/'resultado.json')),flush=True)
