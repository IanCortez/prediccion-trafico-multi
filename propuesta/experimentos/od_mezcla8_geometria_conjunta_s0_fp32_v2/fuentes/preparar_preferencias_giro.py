from common import *
from preferencias_giro import collect_turns
from map_matching import node_connections
from ruteo_compacto import compressed_csr
from scipy.spatial import cKDTree
from numba import set_num_threads
started=time.time();set_num_threads(5);folder=DATA/'preferencias_giro_2014';folder.mkdir(exist_ok=True);assert not (folder/'manifiesto.json').exists()
raw=dict(np.load(DATA/'mapa_dirigido_2014.npz'));comp=dict(np.load(DATA/'grafo_compacto_2014.npz'));D=np.load(DATA/'distancias_2014/D.npy',mmap_mode='r');conn=node_connections(comp);graph,chosen=compressed_csr(comp,'length');routing=(comp['chain_offsets'],comp['flat_incoming_arc'],graph.indptr,graph.indices,chosen,comp['chain_length']);z=dict(np.load(DATA/'trayectorias_train.npz'));match=DATA/'map_matching_2014_v2';assert (match/'manifiesto.json').exists()
nj=len(comp['junction_raw']);nc=len(comp['chain_src']);degree=np.bincount(comp['chain_src'],minlength=nj);outptr=np.r_[0,np.cumsum(degree)];outchains=np.argsort(comp['chain_src'],kind='stable').astype(np.int32);slot=np.empty(nc,np.int32)
for j in range(nj):slot[outchains[outptr[j]:outptr[j+1]]]=np.arange(degree[j])
transition_offsets=np.r_[0,np.cumsum(degree[comp['chain_dst']])];arc_chain=np.full(len(raw['u']),-1,np.int32);heading_in=np.empty((nc,2));heading_out=np.empty_like(heading_in)
for c in range(nc):
    lo,hi=comp['chain_offsets'][c:c+2];arcs=comp['flat_incoming_arc'][lo+1:hi];arc_chain[arcs]=c;nodes=comp['flat_nodes'][lo:hi];heading_out[c]=raw['xy'][nodes[1]]-raw['xy'][nodes[0]];heading_in[c]=raw['xy'][nodes[-1]]-raw['xy'][nodes[-2]]
assert np.all(arc_chain>=0)
destination=z['xy'][z['offsets'][1:]-1].astype(float)*1000;_,destination_node=cKDTree(raw['xy']).query(destination);fold=(z['timestamp']//(7*86400))%5
outputs=collect_turns(z['xy'],z['offsets'],fold,np.load(match/'edge.npy',mmap_mode='r'),np.load(match/'fraction.npy',mmap_mode='r'),np.load(match/'error_m.npy',mmap_mode='r'),raw['u'],raw['v'],raw['length'],D,conn,routing,arc_chain,comp['to_junction'],raw['xy'][comp['junction_raw']],heading_in,heading_out,transition_offsets,slot,destination_node)
local,global_turn,counts,samples,sample_n=outputs;assert counts[:,3].sum()==0 and counts[:,2].sum()==0
sample=np.concatenate([samples[f,:sample_n[f]] for f in range(5)]);np.save(folder/'local.npy',local);np.save(folder/'global.npy',global_turn);np.savez_compressed(folder/'estructura.npz',outptr=outptr,outchains=outchains,transition_offsets=transition_offsets,chain_slot=slot,arc_chain=arc_chain,heading_in=heading_in,heading_out=heading_out);np.savez_compressed(folder/'muestra_oof.npz',events=sample,trip_timestamp=z['timestamp'])
write_json(folder/'manifiesto.json',dict(source_match_sha256=sha(match/'manifiesto.json'),map_sha256=sha(DATA/'mapa_dirigido_2014.npz'),source_trips=len(z['row']),source_gps_points=len(z['xy']),counts_by_fold=counts,count_names=['turn_events','trips_with_turn','sample_overflow','invalid_successor'],sample_n=sample_n,transitions=int(transition_offsets[-1]),local_shape=local.shape,global_shape=global_turn.shape,conditioning='Incoming chain, destination bearing relative to incoming heading (8), Euclidean remaining distance (<500,<1500,<4000,>=4000 m). No arrival time.',quality_weight='max(0.05,(1-max GPS projection error/80)^2)',folds='UTC departure week modulo5; query fold removed before local and shared priors',test_labels_read=False,outputs={name:sha(folder/name) for name in ('local.npy','global.npy','estructura.npz','muestra_oof.npz')},seconds=time.time()-started))
print(dict(events=counts[:,0].sum(),trips=counts[:,1].sum(),sample=len(sample),seconds=time.time()-started),flush=True)
