"""Check recurrentteacher topology/terminal semantics before training; no primarydata change."""
from base_replanteo import *
access=deny_test();start=time.time();corpus=OUT/'datos_calles_parciales_Train_s0';meta=read(corpus/'resultado.json');c=dict(np.load(DATA/'grafo_compacto_2014.npz'))
length=c['chain_length'];selfloop=c['chain_src']==c['chain_dst'];counts=dict(trips=0,tokens=0,known_tokens=0,
 terminal_known_matching=0,terminal_after_gap_or_unknown_entry=0,terminal_distance_inconsistent=0,terminal_selfloop_extra_traversals=0,
 initial_known_matching=0,initial_unknown_or_gap=0,known_full_entry_tokens=0,full_entry_excess_distance=0,
 full_entry_excess_selfloops=0,full_entry_excess_nonselfloops=0,zero_distance_known_tokens=0)
examples=[];max_excess=0.
for item in meta['shards']:
 z=dict(np.load(corpus/item['file']));chain=z['chain'].astype(np.int32);dist=z['distance_m'].astype(np.float64);known=chain!=65535
 counts['trips']+=len(z['row']);counts['tokens']+=len(chain);counts['known_tokens']+=int(known.sum());counts['zero_distance_known_tokens']+=int(np.sum(known&(dist<=1e-8)))
 full=np.zeros(len(chain),bool);good_prev=z['next_route_valid'][:-1];full[1:]=good_prev
 for a in z['offsets'][:-1]:full[a]=False
 selected=np.flatnonzero(full&known);counts['known_full_entry_tokens']+=len(selected)
 excess=dist[selected]-length[chain[selected]];bad=excess>.1
 counts['full_entry_excess_distance']+=int(bad.sum());counts['full_entry_excess_selfloops']+=int(selfloop[chain[selected[bad]]].sum())
 counts['full_entry_excess_nonselfloops']+=int((~selfloop[chain[selected[bad]]]).sum())
 if len(excess):max_excess=max(max_excess,float(excess.max()))
 for i,(a,b) in enumerate(zip(z['offsets'][:-1],z['offsets'][1:])):
  initial_ok=chain[a]==int(z['initial_chain'][i]);counts['initial_known_matching']+=int(initial_ok);counts['initial_unknown_or_gap']+=int(not initial_ok)
  last=b-1;terminal_ok=known[last] and chain[last]==int(z['terminal_chain'][i])
  if not terminal_ok:continue
  counts['terminal_known_matching']+=1
  if last==a and initial_ok:entry=float(z['initial_offset_m'][i])
  elif full[last]:entry=0.
  else:counts['terminal_after_gap_or_unknown_entry']+=1;continue
  expected=float(z['terminal_offset_m'][i])-entry;difference=dist[last]-expected
  if abs(difference)>.1:
   counts['terminal_distance_inconsistent']+=1
   is_loop=bool(selfloop[chain[last]] and difference>0)
   counts['terminal_selfloop_extra_traversals']+=int(is_loop)
   if len(examples)<12:examples.append(dict(row=int(z['row'][i]),chain=int(chain[last]),self_loop=bool(selfloop[chain[last]]),
    entry_m=entry,terminal_offset_m=float(z['terminal_offset_m'][i]),recorded_traversal_m=float(dist[last]),difference_m=float(difference),chain_length_m=float(length[chain[last]])))
result=dict(complete=True,scope='Train-only recurrentteacher semantic audit, no primaryfilter/labelchanges.',counts=counts,
 max_full_entry_excess_m=max_excess,examples=examples,
 issue='Corebankmergesconsecutiveidenticalchaincodes. Selflooprepeatedlapscanbecompressed; before recurrentteacher forcing, expandlosslesslyinDistancewithsum-preservingtimeallocation, ormaskambiguousauxtransition/EOS. NeverforceEOSfromrawtokenindex when physicallyinconsistent.',
 required='KeeporiginalprimaryT for everyquery; unknownGAP clocks retained. Reconstructedsequence labels remain pseudo-observations, not annotatedexactroadpath/time.',
 corpus_manifest_sha256=sha(corpus/'resultado.json'),source_sha256=sha(__file__),
 no_Validation_or_Test_access=True,seconds=time.time()-start)
write(OUT/'auditoria_etiquetas_secuenciales_viales.json',result);print(json.dumps(dict(counts=counts,max_full_entry_excess_m=max_excess,seconds=result['seconds'])),flush=True)
