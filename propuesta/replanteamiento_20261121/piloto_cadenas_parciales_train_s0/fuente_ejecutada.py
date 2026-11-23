"""Bridge only trusted Train map observations; preserve original totalTime and explicit unknown gaps."""
from base_replanteo import *
import shutil
from numba import njit,set_num_threads
from consulta_mapa2014 import resources
from ruteo_proyectado import collect_cost_path
access=deny_test();start=time.time();set_num_threads(2);run=OUT/'piloto_cadenas_parciales_train_s0';assert not run.exists();run.mkdir();shutil.copy2(__file__,run/'fuente_ejecutada.py')
diag=OUT/'diagnostico_resets_viales_s0';proof=read(diag/'resultado.json');assert proof['partial_supervision_step_coverage_gate_passed'] and proof['all_original_matching_outputs_exact']
z=dict(np.load(diag/'candidatos_y_reinicios_Train.npz'));original=dict(np.load(OUT/'piloto_etiquetas_viales_train_s0/etiquetas_observadas_Train.npz'))
assert np.array_equal(z['row'],original['row'])
raw,comp,sample,tree,routes,profiles=resources();set_num_threads(2);D,conn,routing=routes['length']
flat=comp['flat_incoming_arc'];valid=flat>=0;position_chain=np.repeat(np.arange(len(comp['chain_src']),dtype=np.int32),np.diff(comp['chain_offsets']))
edge_chain=np.full(len(raw['u']),-1,np.int32);edge_chain[flat[valid]]=position_chain[valid];assert (edge_chain>=0).all()
matching=z['candidate_edges']==z['matched_edge'][:,None];projection_error=np.where(matching,z['candidate_error_m'],np.inf).min(1)
@njit(cache=True)
def append_token(code,time,distance,used,c,t,d):
 if used>0 and code[used-1]==c:time[used-1]+=t;distance[used-1]+=d;return used
 if used>=len(code):return -1
 code[used]=c;time[used]=t;distance[used]=d;return used+1
@njit(cache=True)
def one_trip(a,b,edge,fraction,flags,error,u,v,length,D,conn,routing,edge_chain):
 cap=max(4096,(b-a)*128);code=np.empty(cap,np.int32);tt=np.empty(cap,np.float64);dd=np.empty(cap,np.float64);used=0
 ids=np.empty(4096,np.int32);parts=np.empty(4096,np.float64);unknown=0;bridgefail=0;stationary=0;overflow=0
 for i in range(a+1,b):
  if flags[i]>0 or error[i-1]>100 or error[i]>100:
   used=append_token(code,tt,dd,used,65535,15.,0.);unknown+=1
  else:
   count,cost=collect_cost_path(edge[i-1],fraction[i-1],edge[i],fraction[i],u,v,length,length,D,conn,routing,ids,parts)
   if count<0 or not np.isfinite(cost) or cost>750.001:
    used=append_token(code,tt,dd,used,65535,15.,0.);unknown+=1;bridgefail+=1
   else:
    total=parts[:count].sum()
    if total<=1e-6:
     used=append_token(code,tt,dd,used,edge_chain[edge[i-1]],15.,0.);stationary+=1
    else:
     for j in range(count):
      if parts[j]<=0:continue
      used=append_token(code,tt,dd,used,edge_chain[ids[j]],15*parts[j]/total,parts[j])
      if used<0:break
  if used<0:overflow=1;break
 if overflow:return np.array([65535],np.uint16),np.array([(b-a-1)*15],np.float32),np.array([0.],np.float32),b-a-1,bridgefail,stationary,overflow
 return code[:used].astype(np.uint16),tt[:used].astype(np.float32),dd[:used].astype(np.float32),unknown,bridgefail,stationary,overflow
codes=[];times=[];distances=[];ptr=[0];unknown=[];fails=[];stationary=[];overflow=[];adjacency=[];max_error=0.
for index,(a,b,target) in enumerate(zip(z['offsets'][:-1],z['offsets'][1:],z['true_duration_s'])):
 c,t,d,n,f,s,o=one_trip(a,b,z['matched_edge'],z['matched_fraction'],z['reset_flags'],projection_error,raw['u'],raw['v'],raw['length'],D,conn,routing,edge_chain)
 assert len(c)>0 and (t>=0).all() and np.isfinite(t).all() and np.isfinite(d).all()
 error=abs(float(t.astype(np.float64).sum())-float(target));max_error=max(max_error,error);assert error<.001
 code32=c.astype(np.int32);known=c!=65535
 mask=np.zeros(len(c),bool)
 if len(c)>1:
  both=known[:-1]&known[1:]
  valid_pair=np.flatnonzero(both)
  mask[valid_pair]=comp['chain_dst'][code32[valid_pair]]==comp['chain_src'][code32[valid_pair+1]]
 # Missing and nonadjacent transitions remain masked auxiliary labels; no primary row is removed.
 codes.append(c);times.append(t);distances.append(d);adjacency.append(mask);ptr.append(ptr[-1]+len(c));unknown.append(n);fails.append(f);stationary.append(s);overflow.append(o)
 if (index+1)%1024==0:print(json.dumps(dict(stage='trusted_chain_labels',trips=index+1,seconds=time.time()-start)),flush=True)
code=np.concatenate(codes);time_s=np.concatenate(times);distance_m=np.concatenate(distances);route_next_mask=np.concatenate(adjacency)
file=run/'cadenas_y_tiempos_Train.npz'
np.savez_compressed(file,row=z['row'],train_index=original['train_index'],timestamp=original['timestamp'],true_duration_s=z['true_duration_s'],
 offsets=np.asarray(ptr,np.int64),chain=code,duration_s=time_s,distance_m=distance_m,auxiliary_next_route_valid=route_next_mask,
 untrusted_step_n=np.asarray(unknown,np.uint8),bridge_failure_n=np.asarray(fails,np.uint8),stationary_step_n=np.asarray(stationary,np.uint8),
 endpoint_public_candidates=original['endpoint_public_candidates'],endpoint_public_fraction=original['endpoint_public_fraction'])
assert file.stat().st_size<200*1024**2
unknown_time=float(time_s[code==65535].astype(np.float64).sum());total=float(z['true_duration_s'].sum());steps=len(z['points_xy'])-len(z['row'])
result=dict(scope='Train-only auxiliary pseudo-label pilot, not primary prediction/evaluation or claimed groundtruth route.',
 trips=len(z['row']),tokens=len(code),mean_tokens_per_trip=len(code)/len(z['row']),
 original_total_time_labels_preserved=True,max_auxiliary_sum_error_s=max_error,
 unknown_time_fraction=unknown_time/total,trusted_time_fraction=1-unknown_time/total,
 bridge_failures=int(sum(fails)),overflow_trips=int(sum(overflow)),stationary_steps=int(sum(stationary)),
 valid_auxiliary_route_transitions=int(route_next_mask.sum()),route_token_gap_id=65535,
 semantics='Each original15secondinterval is distributed over public shortest-path arc lengths and merged into compactchains. Unknown/reset/highprojectionerror intervals keep15seconds as an explicit maskedGAP; stationaryintervals keep15seconds andzero distance. Nonadjacent route targets masked.',
 limitations=['Pseudo-labels infer unobserved paths between15secondGPSsamples; not observed exact streets.','Data-quality sample only for auxiliary preparation; every primary query/label remains unchanged.','Endpoint public projections are query-safe; HMMendpointdirection is a Train auxiliary target, not a query input.','Negative/zero distance and partial-chain traversals require robust time heads; do not treat every token as a full chain.'],
 ready_for_full_model_training=False,next_required='Execute actual neural absorbingOD/batch/singleton proof and Train-onlytime-prediction oracle before full primary streetmodel.',
 source_sha256=sha(run/'fuente_ejecutada.py'),trace_predictions_sha256=sha(diag/'candidatos_y_reinicios_Train.npz'),predictions_sha256=sha(file),
 artifact_bytes=file.stat().st_size,Validation_access=False,Test_access=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(run/'resultado.json',result)
print(json.dumps({k:result[k] for k in ['trips','tokens','trusted_time_fraction','max_auxiliary_sum_error_s','bridge_failures','overflow_trips','stationary_steps','seconds']}),flush=True)
