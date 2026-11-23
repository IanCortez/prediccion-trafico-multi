"""Independent unpack/reconstruction verification of every repairedteacher token; no duration changes."""
from base_replanteo import *
access=deny_test();start=time.time();folder=OUT/'adaptador_progreso_vial_s0';meta=read(folder/'resultado.json');core=OUT/'datos_calles_parciales_Train_s0';cm=read(core/'resultado.json')
assert meta['complete'] and sha(core/'resultado.json')==meta['core_manifest_sha256']
c=dict(np.load(DATA/'grafo_compacto_2014.npz'));max_error=0.;total=0;macro_n=0;legacy_noninitial_wrong=0;initial_nonzero=0;known_time=0.;full_time=0.
for item,origin in zip(meta['files'],cm['shards']):
 file=folder/item['file'];source=core/origin['file'];assert sha(file)==item['sha256'] and sha(source)==item['source_sha256']==origin['sha256']
 a=dict(np.load(file));z=dict(np.load(source));n=len(z['chain']);code=z['chain'].astype(np.int32);safe=np.where(code==65535,0,code)
 known=np.unpackbits(a['position_known_packed'],count=n).astype(bool);macro=np.unpackbits(a['macro_next_valid_packed'],count=n).astype(bool);reset=np.unpackbits(a['reset_before_packed'],count=n).astype(bool)
 entry=np.zeros(n,np.float64);exit=c['chain_length'][safe].copy();ii=a['exception_index'];entry[ii]=a['entry_m'];exit[ii]=a['exit_m']
 assert not np.any(known&(code==65535))
 error=np.abs((exit[known]-entry[known])-z['distance_m'][known]);max_error=max(max_error,float(error.max(initial=0)));assert error.max(initial=0)<.101
 assert np.all(entry[known]>=0) and np.all(exit[known]<=c['chain_length'][safe[known]]+.101)
 starts=z['offsets'][:-1];ends=z['offsets'][1:]-1
 assert reset[starts].all() and not macro[ends].any()
 idx=np.flatnonzero(macro);assert np.all(idx+1<n) and known[idx].all() and known[idx+1].all()
 assert np.all(np.abs(exit[idx]-c['chain_length'][code[idx]])<=.101) and np.all(np.abs(entry[idx+1])<=.101)
 assert np.array_equal(c['chain_dst'][code[idx]],c['chain_src'][code[idx+1]]) and not reset[idx+1].any()
 eos=a['terminal_eos_valid'];last=ends[eos];trip=np.flatnonzero(eos)
 terminal=z['terminal_offset_m'][trip].astype(np.float64);selfloop=c['chain_src'][code[last]]==c['chain_dst'][code[last]]
 terminal=np.where(selfloop&(terminal<=1e-4),c['chain_length'][code[last]],terminal)
 assert known[last].all() and np.array_equal(code[last],z['terminal_chain'][trip]) and np.all(np.abs(exit[last]-terminal)<.101)
 for first,end,target in zip(starts,z['offsets'][1:],z['true_duration_s']):
  assert abs(float(z['duration_s'][first:end].astype(np.float64).sum())-float(target))<.001
 legacy=z['remaining_value_valid']&known&(np.abs(entry)>.1)
 initial_nonzero+=int(legacy[starts].sum());legacy[starts]=False;legacy_noninitial_wrong+=int(legacy.sum())
 total+=n;macro_n+=int(macro.sum());known_time+=float(z['duration_s'][known].astype(np.float64).sum());full_time+=float(z['true_duration_s'].astype(np.float64).sum())
result=dict(passed=True,tokens_verified=total,macro_transitions_verified=macro_n,reconstructed_positions_consistent_to_m=.101,max_distance_error_m=max_error,
 all_macro_edges_actual_chain_end_to_start=True,all_arrival_targets_physicallyconsistent=True,all_original_total_times_preserved=True,
 corrected_legacy_noninitial_zero_entry_error_n=legacy_noninitial_wrong,legacy_initial_nonzero_already_handled_n=initial_nonzero,
 diagnostic_correction='Preparercounter legacy_remaining_wrong_entry includedcorrectlyhandledinitialpartialtokens. Usecorrected_noninitialcount here; no outputposition/maskdata changed.',
 position_known_time_fraction=known_time/full_time,partial_supervision_time_coverage_95percent_gate_passed=bool(known_time/full_time>=.95),
 source_core_arrays_and_shards_unchanged=True,source_sha256=sha(__file__),adapter_manifest_sha256=sha(folder/'resultado.json'),
 no_Validation_or_Test_access=True,seconds=time.time()-start)
assert total==41775036 and macro_n==meta['counts']['macro_next_valid'] and result['partial_supervision_time_coverage_95percent_gate_passed']
write(OUT/'verificacion_adaptador_progreso_vial.json',result)
print(json.dumps({k:result[k] for k in ['passed','tokens_verified','macro_transitions_verified','corrected_legacy_noninitial_zero_entry_error_n','position_known_time_fraction','seconds']}),flush=True)
