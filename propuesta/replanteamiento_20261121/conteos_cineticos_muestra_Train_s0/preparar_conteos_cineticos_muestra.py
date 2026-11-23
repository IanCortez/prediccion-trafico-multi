"""4096FITsourcecount encoding/resourceprobe; no graphheldMAE or modeltraining."""
from base_replanteo import *
import pandas as pd
from conteos_red_cinetica import *
start=time.time();access=deny_test()
run=OUT/'conteos_cineticos_muestra_Train_s0';assert not run.exists();run.mkdir()
parent=OUT/'verificacion_estados_cineticos_Train_s0'
proof=read(parent/'resultado.json');assert proof['passed'] and proof['clock_from_originalGPS_not_road_edgecount']
source=parent/'estados_ajuste4096.npz';assert sha(source)==proof['artifact_sha256']
with np.load(source) as z:
 offsets=z['offsets'];state=z['kinetic_state'];valid=z['valid_auxiliary_pair']
 error=z['error_m'];row=z['row'];timestamp=z['timestamp'];T=z['original_time_s'];parents=z['parent']
n=np.diff(offsets);p=np.repeat(np.arange(len(n)),n);index=np.arange(len(state))-np.repeat(offsets[:-1],n)
point_time=timestamp[p]+15*index
local=pd.to_datetime(point_time,unit='s',utc=True).tz_convert('Europe/Lisbon')
how=np.asarray(local.dayofweek,dtype=np.int16)*24+np.asarray(local.hour,dtype=np.int16)
fold=((timestamp//604800)%5).astype(np.int16)
context=fold[p]*168+how
assert np.all((context>=0)&(context<840)) and np.array_equal(point_time[offsets[1:]-1],timestamp+T)
K=proof['theoretical_kinetic_states'];assert K%3==0 and int(context.max())*K*K+K*K<np.iinfo(np.uint64).max
indices=np.flatnonzero(valid)
assert np.all(p[indices]==p[indices+1]) and np.all(point_time[indices+1]-point_time[indices]==15)
keys,counts=aggregate_keys(encode_pairs(context[indices],state[indices],state[indices+1],K))
end=offsets[1:]-1;end=end[error[end]<=80]
begin=offsets[:-1];begin=begin[(error[begin]<=80)&valid[begin]]
stop_keys,stop_counts=aggregate_keys(encode_events(context[end],state[end],K))
start_keys,start_counts=aggregate_keys(encode_events(context[begin],state[begin],K))
assert int(counts.sum(dtype=np.uint64))==len(indices)
position_keys,position_count=position_counts(keys,counts,K)
assert int(position_count.sum(dtype=np.uint64))==len(indices)
assert len(np.unique(keys))==len(keys) and (counts>0).all()
foldproof=[]
for excluded in range(5):
 selected,weights=select_fold(keys,counts,K,excluded)
 contexts,_,_=decode_pairs(selected,K)
 assert np.all(contexts//168!=excluded)
 original_context=keys//np.uint64(K)**np.uint64(2)
 mutated=counts.astype(np.uint64).copy();mutated[original_context//168==excluded]+=1234
 after,w2=select_fold(keys,mutated,K,excluded)
 assert np.array_equal(selected,after) and np.array_equal(weights,w2)
 # The nestedpositioncontrol must also exclude the wholeparentfold.
 pk,pw=position_counts(selected,weights,K);ctx,_,_=decode_pairs(pk,K//3)
 assert np.all(ctx//168!=excluded)
 foldproof.append(dict(excluded_fold=excluded,remaining_unique_pairs=len(selected),remaining_intervals=int(weights.sum(dtype=np.uint64))))
file=run/'conteos_fuentes4096.npz'
np.savez_compressed(file,kinetic_pair_keys=keys,kinetic_pair_counts=counts,
 stop_keys=stop_keys,stop_counts=stop_counts,start_keys=start_keys,start_counts=start_counts,
 kinetic_state_count=np.array(K,np.int64),source_row=row,source_timestamp=timestamp,source_trueT_s=T,source_parent=parents)
with np.load(OUT/'indices_escala_Train_s0/metadatos_Train.npz') as z:fit=z['train_index'][:int(z['fit_n'])]
prefix_intervals=int(np.load(DATA/'train_y.npy')[fit].sum(dtype=np.float64)/15)
MiB=file.stat().st_size/1024**2
report=dict(passed=True,scope='CompactFITsourcecounts and sourceclock only; probability/kernel/fallback proofs stillpending.',
 sourceparents=len(n),sourcepoints=len(state),original_sourceintervals=int(np.sum(n-1)),valid_auxiliaryintervals=len(indices),
 source_clock15s_exact=True,no_crossparent_or_gap_compression=True,all_originalT_preserved=True,
 public_static_kinetic_state_count=K,unique_kinetic_contextpairs=len(keys),unique_position_contextpairs=len(position_keys),
 start_sourceobservations=int(start_counts.sum(dtype=np.uint64)),stop_sourceobservations=int(stop_counts.sum(dtype=np.uint64)),
 nested_position_andkinetic_totalcounts_equal=True,wholeparentfold_countselection_intervention_passed=foldproof,
 full_kernel_backoff_probability_intervention_pending=True,
 artifact_MiB=MiB,artifact_sha256=sha(file),prefix_intervals=prefix_intervals,
 naive_linear_compressed_scaling_MiB=MiB*prefix_intervals/int(np.sum(n-1)),
 scaling_is_not_a_guarantee='Fullsourcemerge may compress differently; enforce actual350MiBpersistentbudget, do not assume fromsample.',
 no_rawGPSbank_or_populationmapmatchingcache=True,source_dataset_sha256=sha(source),
 queryheld_MAE_computed=False,Validation=False,Test=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start,
 sources={name:sha(OUT/name) for name in ['preparar_conteos_cineticos_muestra.py','conteos_red_cinetica.py']})
write(run/'resultado.json',report)
for name in report['sources']:(run/name).write_bytes((OUT/name).read_bytes())
print((run/'resultado.json').read_text(encoding='utf-8'),flush=True)
