"""Bounded street-label feasibility pilot; decode only explicitly selected eligibleTrain raw rows."""
from base_replanteo import *
import csv,shutil
from numba import set_num_threads
from consulta_mapa2014 import resources,ORIGIN
from map_matching import project_candidates,viterbi_match
access=deny_test();start=time.time();run=OUT/'piloto_etiquetas_viales_train_s0';assert not run.exists();run.mkdir()
shutil.copy2(__file__,run/'fuente_ejecutada.py')
plan=dict(scope='Train-only map-label feasibility, not predictive MAE; primary cohort/labels unchanged.',
 n=4096,seed=20261122,raw_read='Binary row membership in selected eligibleTrain BEFORE text/CSV/POLYLINE decode; no heldout/Test path decoded.',
 sample='EligibleTrain source memory with existing GPS-quality condition, used ONLY for auxiliary label pilot; no primary query removed.',
 matching='Existing frozen public2014map, nearest4projections/Viterbi sigma30m; original15second cadence.',
 source_label='Road edge/fraction and compact street-chain at observedGPSpoints. Bridging between observations is NOT yet reconstructed.',
 gate='At least90percent trips withzeroViterbiresets AND95percentpoints projectedwithin100m; prerequisite only, not time-prediction evidence.',
 resources='CPU2; <=200MiBpilot artifacts; no GPU or whole-dataset trajectory cache.',
 future_requirements=['Pure query endpoint projections from5fields; HMMendpoints labels only.','Expand missing chains and derive sum-preserving auxiliary segmenttimes before actualUNITEadaptation.','Keep original(N−1)*15 primaryduration; fixed/predicted decoding horizon, absorbing terminal states and batch invariance.'],
 Validation=False,Test='sealed')
write(run/'protocolo.json',plan)
m=dict(np.load(OUT/'datos_prototipos_historicos_s0/memoria_Train.npz'));ii=np.sort(np.random.default_rng(20261122).choice(len(m['row']),4096,replace=False))
rows=m['row'][ii];order={int(row):j for j,row in enumerate(rows)};polys=[None]*len(rows)
rawpath=ROOT/'tesis/porto/porto/train.csv';found=0
with rawpath.open('rb') as f:
 header=next(csv.reader([next(f).decode('utf-8-sig').strip()]))
 polygon_col=header.index('POLYLINE');timestamp_col=header.index('TIMESTAMP')
 for rawrow,line in enumerate(f):
  j=order.get(rawrow)
  if j is None:continue
  fields=next(csv.reader([line.decode('utf-8')]))
  assert int(fields[timestamp_col])==int(m['timestamp'][ii[j]])
  points=np.asarray(json.loads(fields[polygon_col]),dtype=np.float64)
  assert points.ndim==2 and points.shape[1]==2 and 5<=len(points)<=120
  assert (len(points)-1)*15==m['time_s'][ii[j]]
  polys[j]=points;found+=1
  if found==len(rows):break
assert found==4096 and all(p is not None for p in polys)
offsets=np.r_[0,np.cumsum([len(p) for p in polys])].astype(np.int64)
ll=np.vstack(polys);del polys
points=np.column_stack([(ll[:,0]-ORIGIN[0])*111195*np.cos(np.deg2rad(ORIGIN[1])),(ll[:,1]-ORIGIN[1])*111195]);del ll
print(json.dumps(dict(stage='selected_Train_only_GPS_decoded',trips=found,points=len(points),seconds=time.time()-start)),flush=True)
raw,comp,sample_edges,tree,routes,profiles=resources();set_num_threads(2)
_,near=tree.query(points,k=32,workers=2)
edges,fraction,error=project_candidates(points,near,sample_edges,raw['xy'],raw['u'],raw['v']);del near
D,conn,routing=routes['length']
best,best_fraction,best_error,resets=viterbi_match(points,offsets,edges,fraction,error,raw['u'],raw['v'],raw['length'],D,*conn)
start_index=offsets[:-1];end_index=offsets[1:]-1;ep=np.column_stack([start_index,end_index]).ravel()
# Same public endpoint projections when no interior query points are supplied.
_,nearest_endpoint=tree.query(points[ep],k=32,workers=2)
pe,pf,pd=project_candidates(points[ep],nearest_endpoint,sample_edges,raw['xy'],raw['u'],raw['v'])
assert np.array_equal(pe,edges[ep]) and np.array_equal(pf,fraction[ep]) and np.array_equal(pd,error[ep])
chain_for_position=np.repeat(np.arange(len(comp['chain_src']),dtype=np.int32),np.diff(comp['chain_offsets']))
flat=comp['flat_incoming_arc'];valid=flat>=0;assert np.unique(flat[valid]).size==len(raw['u'])
edge_to_chain=np.full(len(raw['u']),-1,np.int32);edge_to_chain[flat[valid]]=chain_for_position[valid];assert (edge_to_chain>=0).all()
chains=edge_to_chain[best];observed_lengths=[];adjacent=0;transitions=0;repeats=0
for a,b in zip(offsets[:-1],offsets[1:]):
 c=chains[a:b];seq=c[np.r_[True,c[1:]!=c[:-1]]];observed_lengths.append(len(seq))
 if len(np.unique(seq))<len(seq):repeats+=1
 if len(seq)>1:
  adjacent+=int(np.sum(comp['chain_dst'][seq[:-1]]==comp['chain_src'][seq[1:]]));transitions+=len(seq)-1
zero=float(np.mean(resets==0));near100=float(np.mean(best_error<=100));passed=zero>=.9 and near100>=.95
endpoint_match=(best[ep,None]==edges[ep]).any(-1);assert endpoint_match.all()
endpoint_top1_disagreement=float(np.mean(best[ep]!=edges[ep,0]))
np.savez_compressed(run/'etiquetas_observadas_Train.npz',row=rows,train_index=m['train_index'][ii],timestamp=m['timestamp'][ii],
 true_duration_s=m['time_s'][ii],offsets=offsets,edge=best,edge_fraction=best_fraction.astype(np.float16),
 projection_error_m=best_error.astype(np.float16),chain=chains.astype(np.uint16),resets=resets,
 endpoint_public_candidates=edges[ep].reshape(-1,2,4),endpoint_public_fraction=fraction[ep].reshape(-1,2,4).astype(np.float16))
bytes_=sum(p.stat().st_size for p in run.iterdir() if p.is_file());assert bytes_<200*1024**2
result=dict(passed=passed,trips=found,points=len(points),zero_reset_trip_fraction=zero,point_projection_within100m=near100,
 projection_error_median_m=float(np.median(best_error)),projection_error_p95_m=float(np.quantile(best_error,.95)),
 HMM_endpoint_in_public4_fraction=float(endpoint_match.mean()),HMM_vs_public_nearest1_disagreement_fraction=endpoint_top1_disagreement,
 endpoint_projections_without_interiors_exact=True,all_primary_duration_labels_exact=True,
 observed_collapsed_chain_count_quantiles=np.quantile(observed_lengths,[.5,.9,.95,.99,1]),
 observed_chain_jumps_directly_adjacent_fraction=adjacent/max(transitions,1),trips_revisiting_chain_fraction=repeats/found,
 complete_bridged_road_sequence_not_yet_available=True,
 conclusion='Prerequisitepassed: build Train-onlybridgedchain sequences/times and actualneural ODabsorption tests.' if passed else 'Prerequisitefailed: inspect map/projection failures before any full streetmodel.',
 auxiliary_pilot_only=True,primary_cohort_unchanged=True,Validation_paths_decoded=False,Test_paths_decoded=False,
 raw_selected_row_sha256=hashlib.sha256(rows.tobytes()).hexdigest(),source_sha256=sha(run/'fuente_ejecutada.py'),
 mapper_source_sha256=sha(FROZEN/'map_matching.py'),predictions_sha256=sha(run/'etiquetas_observadas_Train.npz'),
 artifact_bytes=bytes_,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(run/'resultado.json',result)
print(json.dumps({k:result[k] for k in ['passed','trips','points','zero_reset_trip_fraction','point_projection_within100m','HMM_vs_public_nearest1_disagreement_fraction','observed_chain_jumps_directly_adjacent_fraction','seconds']}),flush=True)
