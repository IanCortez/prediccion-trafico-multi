"""Versioned sparse teacherpositions and topology masks; originalcorebank/primarylabels immutable."""
from base_replanteo import *
import shutil
from numba import set_num_threads
from posiciones_secuencia_vial import recover_positions
access=deny_test();start=time.time();set_num_threads(2);run=OUT/'adaptador_progreso_vial_s0';assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
for name in ['preparar_posiciones_progreso_vial.py','posiciones_secuencia_vial.py']:shutil.copy2(OUT/name,run/'fuentes'/name)
proof=read(OUT/'verificacion_transiciones_interiores.json');assert proof['passed'] and proof['confirmed_n']==12
corpus=OUT/'datos_calles_parciales_Train_s0';meta=read(corpus/'resultado.json');c=dict(np.load(DATA/'grafo_compacto_2014.npz'))
counts=dict(trips=0,tokens=0,position_known=0,complex_tokens_masked=0,macro_next_valid=0,legacy_next_valid=0,legacy_false_macro_labels=0,
 legacy_remaining_wrong_entry=0,terminal_eos_valid=0,unsupported_or_uncertain_transitions=0,position_conflicts=0,sparse_exceptions=0)
known_time=0.;total_time=0.;files=[];maximum_distance_error=0.
for sid,item in enumerate(meta['shards']):
 source=corpus/item['file'];assert sha(source)==item['sha256'];z=dict(np.load(source))
 original_times_sha=hashlib.sha256(np.ascontiguousarray(z['duration_s']).tobytes()).hexdigest()
 code=z['chain'].astype(np.int32);distance=z['distance_m'].astype(np.float64)
 entry,exit,known,macro,reset,eos,complex_token,conflicts=recover_positions(code,distance,z['offsets'],z['initial_chain'].astype(np.int32),z['initial_offset_m'],
  z['terminal_chain'].astype(np.int32),z['terminal_offset_m'],c['chain_length'],c['chain_src'],c['chain_dst'],c['chain_offsets'],c['prefix_length'],c['flat_nodes'])
 assert hashlib.sha256(np.ascontiguousarray(z['duration_s']).tobytes()).hexdigest()==original_times_sha
 ix=np.flatnonzero(known);error=np.abs((exit[ix]-entry[ix])-distance[ix]);maximum_distance_error=max(maximum_distance_error,float(error.max()) if len(error) else 0.)
 assert np.isfinite(entry[ix]).all() and np.isfinite(exit[ix]).all() and np.all(entry[ix]>=0) and np.all(exit[ix]<=c['chain_length'][code[ix]]+1e-6) and error.max(initial=0)<.101
 exceptions=ix[(np.abs(entry[ix])>.01)|(np.abs(exit[ix]-c['chain_length'][code[ix]])>.01)]
 # Defaults for ordinaryfullchain tokens are0..chainLength. Only nondefaultpositions are persisted.
 file=run/item['file'];np.savez_compressed(file,exception_index=exceptions.astype(np.int32),entry_m=entry[exceptions].astype(np.float32),exit_m=exit[exceptions].astype(np.float32),
  position_known_packed=np.packbits(known),macro_next_valid_packed=np.packbits(macro),reset_before_packed=np.packbits(reset),
  complex_token_packed=np.packbits(complex_token),terminal_eos_valid=eos,token_n=len(code),trip_n=len(z['row']))
 counts['trips']+=len(z['row']);counts['tokens']+=len(code);counts['position_known']+=int(known.sum());counts['complex_tokens_masked']+=int(complex_token.sum())
 counts['macro_next_valid']+=int(macro.sum());counts['legacy_next_valid']+=int(z['next_route_valid'].sum());counts['legacy_false_macro_labels']+=int(np.sum(z['next_route_valid']&~macro))
 counts['legacy_remaining_wrong_entry']+=int(np.sum(z['remaining_value_valid']&known&(np.abs(entry)>.1)))
 counts['terminal_eos_valid']+=int(eos.sum());counts['position_conflicts']+=int(conflicts);counts['sparse_exceptions']+=len(exceptions)
 counts['unsupported_or_uncertain_transitions']+=int(sum(max(int(b-a)-1,0) for a,b in zip(z['offsets'][:-1],z['offsets'][1:]))-macro.sum())
 known_time+=float(z['duration_s'][known].astype(np.float64).sum());total_time+=float(z['true_duration_s'].astype(np.float64).sum())
 files.append(dict(file=file.name,sha256=sha(file),bytes=file.stat().st_size,source_sha256=item['sha256']))
 if (sid+1)%64==0:print(json.dumps(dict(stage='physical_teacher_adapter',shards=sid+1,trips=counts['trips'],seconds=time.time()-start)),flush=True)
assert counts['trips']==894105 and counts['tokens']==41775036
result=dict(complete=True,scope='Train-only privatepseudo-labelgeometry repair, no primarycohort/labelchanges or new queryinputs.',
 counts=counts,position_known_fraction=counts['position_known']/counts['tokens'],position_known_time_fraction=known_time/total_time,
 max_distance_consistency_error_m=maximum_distance_error,original_time_arrays_unmodified=True,primary_queries_removed=0,
 source_core_unchanged=True,core_manifest_sha256=sha(corpus/'resultado.json'),files=files,
 checkpoint_D_must_be_nonpersistent=True,
 semantics='Recoverentry/exit fromphysicaldistance andfull-chain/observedendpointanchors; nodecrosswalk handlesinteriorreversals. Unsupportedmacroturns masked andprefixreset. Raremergedcyclictraversals masked; totalT/GAPclocks retained. No fictionthat allchains startat0.',
 limitations='Map-matched routes/times remainpseudo-observations. Masks cannotlabel actualidling, routeintent or exactroadtime. Macrodecoder doesnotrepresentallinteriorUturns.',
 source_sha256=sha(run/'fuentes/preparar_posiciones_progreso_vial.py'),helper_sha256=sha(run/'fuentes/posiciones_secuencia_vial.py'),
 artifact_MiB=sum(e['bytes'] for e in files)/1024**2,Validation_access=False,Test_access=False,seconds=time.time()-start)
assert result['artifact_MiB']<100;write(run/'resultado.json',result)
print(json.dumps({k:result[k] for k in ['complete','position_known_fraction','position_known_time_fraction','max_distance_consistency_error_m','artifact_MiB','seconds']}),flush=True)
