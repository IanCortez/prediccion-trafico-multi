from common import *
from map_matching import node_connections,edge_distance
from trazado_vial import collect_edge_path
from ruteo_compacto import compressed_csr
raw=dict(np.load(DATA/'mapa_dirigido_2014.npz'));comp=dict(np.load(DATA/'grafo_compacto_2014.npz'));D=np.load(DATA/'distancias_2014/D.npy',mmap_mode='r');conn=node_connections(comp);graph,chosen=compressed_csr(comp,'length');routing=(comp['chain_offsets'],comp['flat_incoming_arc'],graph.indptr,graph.indices,chosen,comp['chain_length'])
z=np.load(DATA/'trayectorias_train.npz');offsets=z['offsets'];matched=np.load(DATA/'map_matching_2014_v2/edge.npy',mmap_mode='r');fraction=np.load(DATA/'map_matching_2014_v2/fraction.npy',mmap_mode='r');ids=np.empty(4096,np.int32);parts=np.empty(4096,np.float64);max_error=0.;checked=0;stops=0
for trip in np.random.default_rng(184).choice(len(offsets)-1,350,replace=False):
    i=offsets[trip]+1;ea=matched[i];eb=matched[i+1];fa=fraction[i];fb=fraction[i+1];expected=edge_distance(ea,fa,eb,fb,raw['u'],raw['v'],raw['length'],D,*conn)
    if not np.isfinite(expected) or expected>750:continue
    count,distance=collect_edge_path(ea,fa,eb,fb,raw['u'],raw['v'],raw['length'],D,conn,routing,ids,parts);assert count>=0
    total=parts[:count].sum();max_error=max(max_error,abs(total-expected));assert abs(distance-expected)<1e-5 and abs(total-expected)<.1
    assert np.all(parts[:count]>0) and np.all(parts[:count]<=raw['length'][ids[:count]]+.001)
    if count>1:assert np.array_equal(raw['v'][ids[:count-1]],raw['u'][ids[1:count]])
    if count>2:assert np.allclose(parts[1:count-1],raw['length'][ids[1:count-1]],atol=.001)
    checked+=1;stops+=expected<5
assert checked>250
write_json(BASE/'instrumentos/trazado_vial.json',dict(passed=True,training_intervals=checked,stopped_or_nearly_stopped=stops,max_distance_error_m=max_error,arc_continuity=True,partial_arc_lengths_valid=True,test_labels_read=False))
print(dict(intervals=checked,max_error_m=max_error),flush=True)
