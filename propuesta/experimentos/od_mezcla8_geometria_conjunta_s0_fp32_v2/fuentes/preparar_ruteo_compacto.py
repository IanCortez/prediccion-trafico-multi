from common import *
from features import features
from ruteo_compacto import *
from scipy.sparse.csgraph import dijkstra
started=time.time();raw=np.load(DATA/"mapa_dirigido.npz");comp=np.load(DATA/"grafo_compacto.npz")
parts=[dict(np.load(DATA/(s+"_x.npz"))) for s in ("train","validation","test")]
d={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
q=query_map(d,raw,comp);N=len(d["row"])
sources=np.unique(q["o_junction"][q["o_junction"]>=0]);lookup=np.full(len(comp["junction_raw"]),-1,dtype=np.int32);lookup[sources]=np.arange(len(sources))
bounds=np.r_[0,np.cumsum([len(p["row"]) for p in parts])]
f,_=features(d);state=(f[:,15]>=5).astype(np.int64)*6+(f[:,14]//4).astype(np.int64)
fold=np.full(N,5,dtype=np.int8);fold[:bounds[1]]=(d["timestamp"][:bounds[1]]//(7*86400))%5
q["profile_id"]=fold*12+state
np.savez_compressed(DATA/"consultas_ruteo_compacto.npz",**q,sources=sources,source_lookup=lookup,row=d["row"],split_bounds=bounds)
# First check against independent original graph paths on fixed held-out input queries only.
refs=np.concatenate([np.load(DATA/(s+"_rutas.npz"))["X"] for s in ("train","validation","test")])
ix=np.r_[np.random.default_rng(772).choice(N,150,replace=False),np.flatnonzero(refs[:,-1]==0)[:20]]
qq={k:v[ix] for k,v in q.items()}
src=np.unique(qq["o_junction"][qq["o_junction"]>=0]);small_lookup=np.full(len(lookup),-1,dtype=np.int32);small_lookup[src]=np.arange(len(src))
checks={}
for kind in ("length","free"):
    graph,chosen=compressed_csr(comp,kind)
    small=dijkstra(graph,directed=True,indices=src)
    cost,choice=choose_static(qq,small,small_lookup,comp,kind)
    snap=qq["o_snap"]+qq["d_snap"]
    calc=cost+snap/(1 if kind=="length" else (20/3.6))
    valid=np.isfinite(calc)&(qq["o_snap"]<500)&(qq["d_snap"]<500)
    expect=refs[ix,0]*1000 if kind=="length" else refs[ix,1]*600
    diff=np.max(np.abs(calc[valid]-expect[valid]))
    assert diff<.03,(kind,diff)
    assert np.array_equal(valid,refs[ix,-1].astype(bool)),kind
    checks[kind]=dict(max_absolute_error=float(diff),n=int(valid.sum()),connectivity_equal=True)
write_json(BASE/"instrumentos/equivalencia_compresion.json",dict(checks=checks,passed=True,source_queries=len(sources),junctions=len(lookup),test_labels_read=False))
print(dict(stage="equivalence_passed",checks=checks,sources=len(sources),seconds=time.time()-started),flush=True)
for kind in ("length","free"):
    graph,chosen=compressed_csr(comp,kind)
    np.save(DATA/f"compact_{kind}_chosen_chains.npy",chosen)
    prefix=DATA/f"compact_{kind}"
    dist_path=Path(str(prefix)+"_dist.npy");pred_path=Path(str(prefix)+"_pred.npy");done_path=Path(str(prefix)+"_done.npy")
    shape=(len(sources),len(lookup))
    if done_path.exists():
        done=np.load(done_path);dist=np.load(dist_path,mmap_mode="r+");pred=np.load(pred_path,mmap_mode="r+")
    else:
        done=np.zeros(len(sources),bool)
        dist=np.lib.format.open_memmap(dist_path,mode="w+",dtype=np.float32,shape=shape)
        pred=np.lib.format.open_memmap(pred_path,mode="w+",dtype=np.int32,shape=shape)
    for first in range(0,len(sources),64):
        last=min(first+64,len(sources))
        if done[first:last].all():continue
        dd,pp=dijkstra(graph,directed=True,indices=sources[first:last],return_predecessors=True)
        dist[first:last]=dd;pred[first:last]=pp;done[first:last]=True
        if first%512==0 or last==len(sources):
            dist.flush();pred.flush();np.save(done_path,done)
            print(dict(kind=kind,sources=f"{last}/{len(sources)}",seconds=time.time()-started),flush=True)
    cost,choice=choose_static(q,dist,lookup,comp,kind)
    np.savez_compressed(DATA/f"compact_{kind}_choices.npz",cost=cost.astype(np.float32),choice=choice)
write_json(DATA/"ruteo_compacto_manifiesto.json",dict(n_queries=N,sources=len(sources),junctions=len(lookup),graph_sha256=sha(DATA/"grafo_compacto.npz"),queries_sha256=sha(DATA/"consultas_ruteo_compacto.npz"),static_path_metrics=["shortest geometry","fastest OSM free flow"],equivalence=checks,seconds=time.time()-started))
print(dict(stage="done",seconds=time.time()-started),flush=True)

