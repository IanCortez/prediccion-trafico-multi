"""CPU-only independent all-token coverage and actualcheckpoint-size audit during training."""
from base_replanteo import *
import torch
from datos_progreso_recurrente import CorrectedSequences
access=deny_test();start=time.time();run=OUT/'progreso_vial_recurrente_s0'
for n in ['entrenar_progreso_recurrente.py','modelo_progreso_recurrente.py','datos_progreso_recurrente.py','modelo_aux_subviajes.py','base_replanteo.py']:
 assert sha(OUT/n)==sha(run/'fuentes'/n)
# Read only: no CUDA allocations / no modification of the running process.
checkpoint=torch.load(run/'reanudacion.pt',map_location='cpu',weights_only=False)
state=checkpoint['state'];assert 'D' not in state and 'static' not in state and 'neighbors' not in state
statebytes=sum(v.numel()*v.element_size() for v in state.values());assert statebytes<50*1024**2
epoch=int(checkpoint['epoch']);del checkpoint,state
teacher=CorrectedSequences()
f=teacher.fragments;n=len(teacher.code)
difference=np.zeros(n+1,np.int32)
np.add.at(difference,f[:,0],1);np.add.at(difference,f[:,0]+f[:,1],-1)
coverage=difference.cumsum(dtype=np.int32)[:-1];del difference
assert coverage.max()==1 and coverage.min()==0
core=OUT/'datos_calles_parciales_Train_s0';adapter=OUT/'adaptador_progreso_vial_s0'
cm=read(core/'resultado.json');am=read(adapter/'resultado.json')
g=dict(np.load(OUT/'banco_valor_vial_s0/grafo_estatico.npz'))
cursor=0;known_total=0;valid_macro=0;valid_eos=0;max_clock_error=0.;covered_parents=0;dest_unknown=0
parent_seen=np.zeros(1023529,bool);parent_seen[f[:,2]]=True
for info,ai in zip(cm['shards'],am['files']):
 z=dict(np.load(core/info['file']));q=dict(np.load(adapter/ai['file']));m=len(z['chain'])
 known=np.unpackbits(q['position_known_packed'],count=m).astype(bool)
 macro=np.unpackbits(q['macro_next_valid_packed'],count=m).astype(bool)
 dest_valid=z['terminal_chain']<53766
 expected_known=known&np.repeat(dest_valid,np.diff(z['offsets']))
 assert np.array_equal(coverage[cursor:cursor+m],expected_known.astype(np.int32))
 indices=np.flatnonzero(macro)
 if len(indices):
  targets=teacher.route[cursor+indices]
  assert np.all((targets>=0)&(targets<6))
  assert np.array_equal(g['neighbors'][z['chain'][indices],targets],z['chain'][indices+1])
 terminal_indices=z['offsets'][1:]-1
 assert np.array_equal(teacher.route[cursor+terminal_indices]==6,q['terminal_eos_valid'])
 for j,(a,b) in enumerate(zip(z['offsets'][:-1],z['offsets'][1:])):
  ti=int(z['train_index'][j]);dt=z['duration_s'][a:b].astype(np.float64)
  elapsed=np.r_[0.,np.cumsum(dt)[:-1]]
  error=np.abs(teacher.elapsed[cursor+a:cursor+b].astype(np.float64)-elapsed).max(initial=0)
  max_clock_error=max(max_clock_error,float(error));assert error<.0002
  assert np.max(np.abs(teacher.remaining[cursor+a:cursor+b].astype(np.float64)-(float(z['true_duration_s'][j])-elapsed)))<.0002
  assert teacher.ts[ti]==z['timestamp'][j] and teacher.y[ti]==z['true_duration_s'][j]
 known_total+=int(expected_known.sum());valid_macro+=len(indices);valid_eos+=int(q['terminal_eos_valid'].sum())
 covered_parents+=int(parent_seen[z['train_index']].sum());dest_unknown+=int((~dest_valid).sum());cursor+=m
assert cursor==41775036 and known_total==int(coverage.sum()) and valid_macro==39413092
eligible=teacher.eligible
# Independently verify each epoch's primary chunk partition without importing/running the trainer.
rng=np.random.default_rng(3001);length=f[:,1];batches=[]
for lo,hi in [(0,16),(16,32),(32,64),(64,128)]:
 ids=np.flatnonzero((length>lo)&(length<=hi));rng.shuffle(ids)
 batches.extend([ids[a:a+512] for a in range(0,len(ids),512)])
rng.shuffle(batches);primary=eligible.copy();rng.shuffle(primary)
chunks=np.array_split(primary,len(batches))
assert np.array_equal(np.sort(np.concatenate(chunks)),np.sort(eligible)) and len(np.unique(primary))==1023485
assert np.array_equal(np.sort(np.concatenate(batches)),np.arange(len(f)))
result=dict(passed=True,no_CUDA_allocations=True,actual_checkpoint_epoch_observed=epoch,state_MiB=statebytes/1024**2,
 public_D_graph_not_in_actual_checkpoint=True,auxiliary_fragments=len(f),auxiliary_parents_with_supervision=covered_parents,
 source_parents_with_unknown_destination=dest_unknown,source_tokens=cursor,known_position_tokens_used_once_per_epoch=known_total,
 known_tokens_coverage_maximum=1,all_unknown_or_unsupported_tokens_masked=True,
 corrected_macro_targets_exact=valid_macro,corrected_EOS_targets_exact=valid_eos,
 original_clock_includes_all_source_token_times=True,max_clock_F32_elapsed_storage_error_s=max_clock_error,
 all_original_primary_queries_once_per_epoch=1023485,all_auxiliary_fragments_once_per_epoch=True,
 source_sha256=sha(__file__),no_Validation_labels_or_Test=True,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(OUT/'verificacion_cobertura_progreso_recurrente.json',result);print(json.dumps(result),flush=True)
