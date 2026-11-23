"""Trace existing Viterbi resets without changing matching, primary labels or query inputs."""
from base_replanteo import *
import inspect,importlib.util
pilot_path=OUT/'piloto_etiquetas_viales_train.py'
assert sha(pilot_path)==sha(OUT/'piloto_etiquetas_viales_train_s0/fuente_ejecutada.py')
prefix=pilot_path.read_text(encoding='utf-8-sig')
needle="best,best_fraction,best_error,resets=viterbi_match("
prefix=prefix[:prefix.index(needle)]
prefix=prefix.replace("run=OUT/'piloto_etiquetas_viales_train_s0'","run=OUT/'diagnostico_resets_viales_s0'")
exec(compile(prefix,str(pilot_path),'exec'),globals())
from map_matching import viterbi_match as legacy_match
body=inspect.getsource(legacy_match.py_func)
body=body.replace('@njit(cache=True,parallel=True)\n','').replace('def viterbi_match(','def viterbi_trace(')
patches=[
 ('n=len(points);best_edges=','n=len(points);reset_flags=np.zeros(n,np.int8);best_edges='),
 ('resets[trip]+=1;new=-.5','reset_flags[i]=2;resets[trip]+=1;new=-.5'),
 ('elif gps>500:resets[trip]+=1','elif gps>500:resets[trip]+=1;reset_flags[i]=1'),
 ('return best_edges,best_fraction,best_error,resets','return best_edges,best_fraction,best_error,resets,reset_flags')]
for old,new in patches:assert body.count(old)==1,old;body=body.replace(old,new)
generated=OUT/'_trazador_viterbi_20261122.py';assert not generated.exists()
generated.write_text('from base_replanteo import np\nfrom numba import njit,prange\nfrom map_matching import edge_distance\n@njit(cache=True,parallel=True)\n'+body,encoding='utf-8')
spec=importlib.util.spec_from_file_location('trazador_viterbi_20261122',generated);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
be,bf,ber,reset,flags=module.viterbi_trace(points,offsets,edges,fraction,error,raw['u'],raw['v'],raw['length'],D,*conn)
old=dict(np.load(OUT/'piloto_etiquetas_viales_train_s0/etiquetas_observadas_Train.npz'))
assert np.array_equal(be,old['edge']) and np.array_equal(bf.astype(np.float16),old['edge_fraction']) and np.array_equal(ber.astype(np.float16),old['projection_error_m']) and np.array_equal(reset,old['resets'])
for a,b,count in zip(offsets[:-1],offsets[1:],reset):assert flags[a]==0 and int(np.count_nonzero(flags[a:b]))==count
bad=np.flatnonzero(flags>0);same_edge_backward_noise=0
for i in bad:
 eligible=False
 for a in range(4):
  for b in range(4):
   if edges[i-1,a]!=edges[i,b]:continue
   backward=(fraction[i-1,a]-fraction[i,b])*raw['length'][edges[i,b]]
   if 0<backward<=30 and error[i-1,a]<=50 and error[i,b]<=50:eligible=True
 same_edge_backward_noise+=int(eligible)
steps=len(points)-len(offsets)+1;trusted=1-len(bad)/steps
np.savez_compressed(run/'candidatos_y_reinicios_Train.npz',row=rows,offsets=offsets,points_xy=points,candidate_edges=edges,
 candidate_fraction=fraction,candidate_error_m=error,matched_edge=be,matched_fraction=bf,reset_flags=flags,true_duration_s=m['time_s'][ii])
result=dict(scope='Diagnostic only; original matching reproduced exactly, no primary label/filter change.',
 trips=len(rows),steps=steps,reset_steps=len(bad),trusted_step_fraction=trusted,
 fraction_trips_without_reset=float(np.mean(reset==0)),
 reset_due_GPS_over500m=int(np.sum(flags==1)),reset_due_no_finite_transition=int(np.sum(flags==2)),
 reset_with_possible_small_backward_same_edge_candidates_fraction=same_edge_backward_noise/max(len(bad),1),
 caveat='Candidate-only GPS-jitter explanation does not prove the previous candidate had finite Viterbi score; no tolerance was changed.',
 whole_trip_quality_gate_remains_failed=True,
 next_question='Can masked partial-route supervision use >=95percenttrustedsteps while retaining every primary query and original totalduration?',
 partial_supervision_step_coverage_gate_passed=bool(trusted>=.95),
 all_original_matching_outputs_exact=True,unknown_step_time_fraction_if_masked=1-trusted,
 route_bridging_not_yet_done=True,source_sha256=sha(__file__),generated_trace_sha256=sha(generated),
 predictions_sha256=sha(run/'candidatos_y_reinicios_Train.npz'),
 Validation_paths_decoded=False,Test_paths_decoded=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(run/'resultado.json',result);write(run/'protocolo.json',dict(scope=result['scope'],primary='Every query and originalduration retained; partialsupervision is a new proposal, not a reinterpretation that the oldgatepassed.',changes='Traceflags only; no matching parameter or filtering change.'))
print(json.dumps({k:result[k] for k in ['trips','steps','reset_steps','trusted_step_fraction','fraction_trips_without_reset','reset_with_possible_small_backward_same_edge_candidates_fraction','partial_supervision_step_coverage_gate_passed','seconds']}),flush=True)
