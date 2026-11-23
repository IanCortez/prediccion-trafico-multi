"""Confirm whether terminalpartial-distance discrepancies are legal rawnode reversals inside macrochains."""
from base_replanteo import *
c=dict(np.load(DATA/'grafo_compacto_2014.npz'));audit=read(OUT/'auditoria_etiquetas_secuenciales_viales.json');examples=audit['examples'];wanted={q['row']:q for q in examples};out=[]
meta=read(OUT/'datos_calles_parciales_Train_s0/resultado.json')
for item in meta['shards']:
 if not wanted:break
 z=dict(np.load(OUT/'datos_calles_parciales_Train_s0'/item['file']))
 for i,row in enumerate(z['row']):
  if int(row) not in wanted:continue
  e=wanted.pop(int(row));a,b=z['offsets'][i:i+2];last=b-1;cur=int(z['chain'][last]);prev=int(z['chain'][last-1]) if last>a else -1
  entry=float(z['terminal_offset_m'][i])-float(z['distance_m'][last]);lo,hi=c['chain_offsets'][cur:cur+2];prefix=c['prefix_length'][lo:hi]
  pos=int(np.argmin(np.abs(prefix-entry)));node=int(c['flat_nodes'][lo+pos]);error=float(abs(prefix[pos]-entry));previous_contains=False;prev_position=None
  if prev>=0 and prev!=65535:
   p,q=c['chain_offsets'][prev:prev+2];hit=np.flatnonzero(c['flat_nodes'][p:q]==node);previous_contains=bool(len(hit))
   if len(hit):prev_position=float(c['prefix_length'][p+hit[0]])
  out.append(dict(row=int(row),current_chain=cur,previous_chain=prev,inferred_entry_m=entry,
   interior_rawnode=int(node),nearest_prefix_error_m=error,entry_is_interior=bool(pos>0 and pos<len(prefix)-1),
   previous_chain_contains_same_node=previous_contains,previous_position_m=prev_position,
   reversed_junction_endpoints=bool(prev>=0 and prev!=65535 and c['chain_src'][prev]==c['chain_dst'][cur] and c['chain_dst'][prev]==c['chain_src'][cur])))
assert len(out)==len(examples)
confirmed=[e for e in out if e['nearest_prefix_error_m']<.1 and e['entry_is_interior'] and e['previous_chain_contains_same_node']]
r=dict(passed=bool(len(confirmed)>0),examples=out,confirmed_n=len(confirmed),
 finding='Compactchainendpointadjacencyalone doesnotimply a full-chaintraversal: rawroadgraph permits switching/reversingatsharedinteriornodes. This is a teacherstate abstractionissue, not wrongdistanceunits.',
 next='Recoverphysicalentry/exitpositions fromdistance+knownanchors; maskunsupportedmacro transitions or representrawnode turns. KeepallprimaryT/queries.',
 source_sha256=sha(__file__),no_Validation_or_Test_access=True)
write(OUT/'verificacion_transiciones_interiores.json',r);print(json.dumps(dict(passed=r['passed'],confirmed_n=len(confirmed),examples=out[:3])),flush=True)
