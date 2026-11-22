from common import *
from features import *
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from geometria_ruteada import sample_routes
from numba import set_num_threads
import argparse
p=argparse.ArgumentParser();p.add_argument('--year',choices=['2014','2026'],required=True);a=p.parse_args();started=time.time();set_num_threads(3)
raw=dict(np.load(DATA/f'mapa_dirigido_{a.year}.npz'));allx=[dict(np.load(DATA/(s+'_x.npz'))) for s in ('train','validation','test')];d={key:np.concatenate([part[key] for part in allx]) for key in allx[0]};F=features(d)[0]
N=len(F);n=len(raw['xy']);E=len(raw['u']);tree=cKDTree(raw['xy']);
def metric(tag):return np.column_stack([(d[tag+'_lon']-ORIGIN[0])*111195*np.cos(np.deg2rad(ORIGIN[1])),(d[tag+'_lat']-ORIGIN[1])*111195])
os,oi=tree.query(metric('o'));ds,di=tree.query(metric('d'));sources,inv=np.unique(oi,return_inverse=True);order=np.argsort(inv,kind='stable');bounds=np.searchsorted(inv[order],np.arange(len(sources)+1));snaps=os+ds
pair=raw['u'].astype(np.int64)*n+raw['v'];chosen=np.argsort(pair);assert len(np.unique(pair))==E
graphs=[csr_matrix((raw[key],(raw['u'],raw['v'])),shape=(n,n)) for key in ('length','seconds')]
flat=np.column_stack([raw['u'],raw['v']]).ravel().astype(np.int32);offsets=np.arange(E+1,dtype=np.int32)*2;junctions=np.arange(n,dtype=np.int32)
folder=DATA/('ruteo_mapa_'+a.year);folder.mkdir(exist_ok=True);chunks=int(np.ceil(len(sources)/64));meta=folder/'config.json'
if meta.exists():
    previous=json.loads(meta.read_text());assert previous['graph_sha256']==sha(DATA/f'mapa_dirigido_{a.year}.npz') and previous['manifest_sha256']==sha(DATA/'manifiesto.json');X=np.load(folder/'X_raw.npy',mmap_mode='r+');done=np.load(folder/'done.npy',mmap_mode='r+')
else:
    X=np.lib.format.open_memmap(folder/'X_raw.npy',mode='w+',dtype=np.float32,shape=(N,49));X[:]=np.nan;done=np.lib.format.open_memmap(folder/'done.npy',mode='w+',dtype=bool,shape=(chunks,));done[:]=False
    write_json(meta,dict(graph_sha256=sha(DATA/f'mapa_dirigido_{a.year}.npz'),manifest_sha256=sha(DATA/'manifiesto.json'),sources=len(sources),chunks=chunks,raw_columns='First two raw columns are meters and seconds until final transformation; remaining shape columns are final.'))
max_length_error=0.;max_cost_error=0.
for chunk,first in enumerate(range(0,len(sources),64)):
    if done[chunk]:continue
    last=min(first+64,len(sources));ids=order[bounds[first]:bounds[last]];root=sources[first:last];lookup=np.full(n,-1,np.int32);lookup[root]=np.arange(len(root));oj=np.column_stack([oi[ids],np.full(len(ids),-1)]);dj=np.column_stack([di[ids],np.full(len(ids),-1)]);negative=np.full_like(oj,-1)
    for which,graph in enumerate(graphs):
        distances,pred=dijkstra(graph,directed=True,indices=root,return_predecessors=True);values=distances[inv[ids]-first,di[ids]];good=np.isfinite(values)&(os[ids]<500)&(ds[ids]<500);choice=np.where(np.isfinite(values),0,-1).astype(np.int8)
        shape=sample_routes(raw['xy'],flat,offsets,np.zeros(1),junctions,graph.indptr,graph.indices,chosen,pred,lookup,choice,negative,negative,negative,negative,oj,dj,F[ids,:2].astype(float)*1000,F[ids,2:4].astype(float)*1000)
        shape[~good]=0;shape[~good,14]=F[ids[~good],6];X[ids,7+which*21:28+which*21]=shape
        if which==0:
            X[ids,0]=np.where(good,values+snaps[ids],d['od_m'][ids])
            if good.any():max_length_error=max(max_length_error,float(np.max(np.abs(shape[good,14]*1000-X[ids[good],0]))));assert max_length_error<.02
        else:X[ids,1]=np.where(good,values+snaps[ids]/(20/3.6),d['od_m'][ids]/(25/3.6))
        # Independent Python summation of predecessor-arc costs, sampled per batch.
        for qi in np.flatnonzero(good)[:1]:
            sr=inv[ids[qi]]-first;node=di[ids[qi]];source=root[sr];cost=0.;steps=0
            while node!=source:
                parent=pred[sr,node];assert parent>=0;edge=graph.indptr[parent]+np.flatnonzero(graph.indices[graph.indptr[parent]:graph.indptr[parent+1]]==node)[0];cost+=graph.data[edge];node=parent;steps+=1;assert steps<n
            max_cost_error=max(max_cost_error,abs(cost-values[qi]));assert max_cost_error<1e-5
    X.flush();done[chunk]=True;done.flush()
    if chunk%20==0 or last==len(sources):print(dict(year=a.year,sources=f'{last}/{len(sources)}',seconds=time.time()-started),flush=True)
assert done.all();result=np.array(X);length=result[:,0].copy();seconds=result[:,1].copy();good=result[:,27]>0
result[:,:7]=np.column_stack([length/1000,seconds/600,np.log(np.maximum(length/d['od_m'],.01)),np.log(np.maximum(seconds*1000/np.maximum(length,1),.01)),os/1000,ds/1000,good.astype(float)])
prefix='archive'+a.year;names=[prefix+'_'+v for v in ('network_shortest_km','network_fastest_time_600s','network_circuity_log','network_pace_log','o_snap_km','d_snap_km','route_available')]
for kind in ('length','free'):names += [prefix+'_'+kind+'_shape_'+str(i) for i in range(14)]+[prefix+'_'+kind+'_'+v for v in ('polyline_km','polyline_circuity_log','turn_abs_pi','turn_signed_pi','sharp_turn_count','max_turn_pi','shape_available')]
assert np.isfinite(result).all();start=0;hashes={}
for split,part in zip(('train','validation','test'),allx):
    end=start+len(part['row']);path=DATA/(split+'_mapa_'+a.year+'.npz');assert not path.exists(),path;np.savez_compressed(path,row=part['row'],X=result[start:end],names=np.array(names));hashes[split]=sha(path);start=end
write_json(DATA/f'mapa_{a.year}_features.json',dict(year=a.year,source_graph_sha256=sha(DATA/f'mapa_dirigido_{a.year}.npz'),features=names,hashes=hashes,query_count=N,route_available_fraction=float(good.mean()),max_length_error_m=max_length_error,max_raw_cost_trace_error=max_cost_error,test_labels_read=False,method='Identical raw directed routing and nine-point polyline features for matched annual Geofabrik extracts. No observed query trajectory.',seconds=time.time()-started))
print(dict(stage='done',year=a.year,available=float(good.mean()),seconds=time.time()-started),flush=True)
