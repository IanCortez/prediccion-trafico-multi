from common import *
from compresion_vial import compress_graph
from ruteo_compacto import compressed_csr,query_map,choose_static
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
import argparse
p=argparse.ArgumentParser();p.add_argument('--year',choices=['2014','2026'],default='2014');p.add_argument('--metric',choices=['length','free'],default='length');a=p.parse_args();started=time.time();path=DATA/f'mapa_dirigido_{a.year}.npz';raw=dict(np.load(path));output=DATA/f'grafo_compacto_{a.year}.npz'
if output.exists():comp=dict(np.load(output))
else:
    comp=compress_graph(raw);np.savez_compressed(output,**comp)
n=len(comp['junction_raw']);graph,chosen=compressed_csr(comp,a.metric);folder=DATA/('distancias_'+a.year+('_free' if a.metric=='free' else ''));folder.mkdir(exist_ok=True);matrix=folder/'D.npy';state=folder/'done.npy'
if state.exists():D=np.load(matrix,mmap_mode='r+');done=np.load(state,mmap_mode='r+');assert D.shape==(n,n)
else:
    D=np.lib.format.open_memmap(matrix,mode='w+',dtype=np.float32,shape=(n,n));done=np.lib.format.open_memmap(state,mode='w+',dtype=bool,shape=(n,));done[:]=False
print(dict(stage='compact',year=a.year,junctions=n,chains=len(comp['chain_src']),matrix_gb=n*n*4/1e9,seconds=time.time()-started),flush=True)
for first in range(0,n,64):
    last=min(first+64,n)
    if done[first:last].all():continue
    rows=dijkstra(graph,directed=True,indices=np.arange(first,last),return_predecessors=False);D[first:last]=rows;D.flush();done[first:last]=True;done.flush()
    if first%1024==0 or last==n:print(dict(stage='all_pairs',sources=f'{last}/{n}',seconds=time.time()-started),flush=True)
# Independent raw-graph check for junction distances, including disconnections.
rg=csr_matrix((raw['length' if a.metric=='length' else 'seconds'],(raw['u'],raw['v'])),shape=(len(raw['xy']),)*2);rng=np.random.default_rng(794);src=rng.choice(n,min(12,n),replace=False);target=rng.choice(n,min(70,n),replace=False);exact=dijkstra(rg,directed=True,indices=comp['junction_raw'][src],return_predecessors=False)[:,comp['junction_raw'][target]];calc=np.asarray(D[np.ix_(src,target)]);assert np.array_equal(np.isfinite(calc),np.isfinite(exact));valid=np.isfinite(exact);error=float(np.max(np.abs(calc[valid]-exact[valid])));assert error<.1
write_json(folder/'manifiesto.json',dict(year=a.year,metric=a.metric,units='m' if a.metric=='length' else 's',raw_graph_sha256=sha(path),compact_graph_sha256=sha(output),distance_sha256=sha(matrix),junctions=n,chains=len(comp['chain_src']),all_raw_arcs_exactly_once=True,independent_pairs=int(exact.size),max_distance_error_m=error,purpose='Static road distances for training-only GPS map matching; no query time labels.',test_labels_read=False,seconds=time.time()-started))
print(dict(stage='done',error_m=error,seconds=time.time()-started),flush=True)


