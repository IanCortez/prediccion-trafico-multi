from common import *
from features import *
from ruteo_compacto import *
from perfiles_ruta import *
from numba import set_num_threads
started=time.time();set_num_threads(4)
comp=np.load(DATA/"grafo_compacto.npz");q=dict(np.load(DATA/"consultas_ruteo_compacto.npz"))
N=len(q["row"]);sources=q["sources"];lookup=q["source_lookup"]
arc=np.load(DATA/"arcos_tiempos_gps.npz")
pfile=DATA/"prefijos_movimiento_compacto.npz"
if pfile.exists():
    p=np.load(pfile);pt,pc,ct,cc=[p[k] for k in ("prefix_time","prefix_support","chain_time","chain_support")]
else:
    pt,pc,ct,cc=chain_profiles(comp["chain_offsets"],comp["flat_incoming_arc"],np.ascontiguousarray(arc["time_s"].T),np.ascontiguousarray(arc["observed_length_m"].T))
    np.savez_compressed(pfile,prefix_time=pt,prefix_support=pc,chain_time=ct,chain_support=cc)
print(dict(stage="prefixes",seconds=time.time()-started),flush=True)
outputs=[];names=[];checks={}
profile=q["profile_id"];snap=q["o_snap"]+q["d_snap"]
for kind in ("length","free"):
    graph,chosen=compressed_csr(comp,kind)
    pred=np.load(DATA/f"compact_{kind}_pred.npy",mmap_mode="r")
    choice=np.load(DATA/f"compact_{kind}_choices.npz")["choice"]
    normal=(choice>=0)&(choice<4);ids=np.flatnonzero(normal)
    oc=np.maximum(choice,0)//2;dc=np.maximum(choice,0)%2
    source_row=np.full(N,-1,dtype=np.int32);dest=np.full(N,-1,dtype=np.int32)
    source_row[ids]=lookup[q["o_junction"][ids,oc[ids]]];dest[ids]=q["d_junction"][ids,dc[ids]]
    order=ids[np.argsort(source_row[ids],kind="stable")]
    bounds=np.searchsorted(source_row[order],np.arange(len(sources)+1))
    result=routed_profiles(pred,sources,graph.indptr,graph.indices,chosen,ct,cc,comp["chain_length"],comp["chain_free"],order,bounds,dest,profile,N)
    for o in range(2):
        for d in range(2):
            ix=np.flatnonzero(choice==o*2+d)
            for tag,slot in (("o",o),("d",d)):
                cid=q[tag+"_chain"][ix,slot];pos=q[tag+"_pos"][ix,slot];valid=cid>=0
                use=ix[valid];c=cid[valid];p=pos[valid];pr=profile[use]
                result[use,0]+=(ct[c,pr]-pt[p,pr]) if tag=="o" else pt[p,pr]
                result[use,1]+=(cc[c,pr]-pc[p,pr]) if tag=="o" else pc[p,pr]
            result[ix,2]+=q["o_length"][ix,o]+q["d_length"][ix,d]
            result[ix,3]+=q["o_free"][ix,o]+q["d_free"][ix,d]
            ix=np.flatnonzero(choice==4+o*2+d);op=q["o_pos"][ix,o];dp=q["d_pos"][ix,d];pr=profile[ix]
            result[ix,0]=pt[dp,pr]-pt[op,pr];result[ix,1]=pc[dp,pr]-pc[op,pr]
            result[ix,2]=comp["prefix_length"][dp]-comp["prefix_length"][op]
            result[ix,3]=comp["prefix_free"][dp]-comp["prefix_free"][op]
    result[:,0]+=snap/(20/3.6);result[:,2]+=snap;result[:,3]+=snap/(20/3.6)
    # Independent raw-arc trace on selected paths, not reusing the profile DP.
    errors=[]
    for i in np.random.default_rng(381).choice(np.flatnonzero(choice>=0),40,replace=False):
        c=int(choice[i]);o=(c%4)//2;d=c%2;arcs=[]
        def add_chain(chain,start=None,end=None):
            lo=comp["chain_offsets"][chain]+1 if start is None else start+1
            hi=comp["chain_offsets"][chain+1] if end is None else end+1
            arcs.extend(comp["flat_incoming_arc"][lo:hi].tolist())
        if c>=4:
            chain=q["o_chain"][i,o];add_chain(chain,q["o_pos"][i,o],q["d_pos"][i,d])
        else:
            chain=q["o_chain"][i,o]
            if chain>=0:add_chain(chain,q["o_pos"][i,o],None)
            root=q["o_junction"][i,o];node=q["d_junction"][i,d];row=lookup[root]
            while node!=root:
                parent=pred[row,node];assert parent>=0
                edge=np.flatnonzero(graph.indices[graph.indptr[parent]:graph.indptr[parent+1]]==node)[0]+graph.indptr[parent]
                add_chain(chosen[edge]);node=parent
            chain=q["d_chain"][i,d]
            if chain>=0:add_chain(chain,None,q["d_pos"][i,d])
        actual=arc["time_s"][profile[i],np.asarray(arcs,dtype=int)].astype(np.float64).sum()+snap[i]/(20/3.6)
        errors.append(abs(actual-result[i,0]))
    assert max(errors)<.1,(kind,max(errors))
    good=(choice>=0)&(q["o_snap"]<500)&(q["d_snap"]<500)&np.isfinite(result).all(1)
    # Missing routes receive a declared physical fallback, not observed query time.
    allx=[dict(np.load(DATA/(s+"_x.npz"))) for s in ("train","validation","test")]
    od=np.concatenate([x["od_m"] for x in allx])
    result[~good,0]=60+od[~good]/(20/3.6);result[~good,1]=0
    result[~good,2]=od[~good];result[~good,3]=od[~good]/(25/3.6)
    confidence=np.clip(result[:,1]/np.maximum(result[:,2],1),0,1)
    X=np.column_stack([result[:,0]/600,result[:,2]/1000,result[:,3]/600,confidence,np.log(np.maximum(result[:,0]/np.maximum(result[:,3],1),.01)),np.log(np.maximum(1000*result[:,0]/np.maximum(result[:,2],1),.01)),good.astype(float)]).astype(np.float32)
    outputs.append(X);names.extend([kind+"_"+s for s in ("gps_time_600s","path_km","freeflow_600s","support_fraction","congestion_log","pace_log","available")])
    checks[kind]=dict(raw_trace_max_error_s=max(errors),route_available=float(good.mean()))
    print(dict(stage=kind,checks=checks[kind],seconds=time.time()-started),flush=True)
X=np.column_stack(outputs)
assert np.isfinite(X).all()
for i,s in enumerate(("train","validation","test")):
    lo,hi=q["split_bounds"][i:i+2]
    np.savez_compressed(DATA/(s+"_gps_rutas.npz"),row=q["row"][lo:hi],X=X[lo:hi],names=np.array(names))
write_json(DATA/"gps_rutas_manifiesto.json",dict(n_features=len(names),profile_sha256=sha(DATA/"arcos_tiempos_gps.npz"),query_sha256=sha(DATA/"consultas_ruteo_compacto.npz"),route_checks=checks,training_profiles="Exclude query UTC departure-week fold; validation/test use training-only full profile",inference="OD, departure and static road map only; chosen route is calculated, not observed",test_labels_read=False,seconds=time.time()-started))
print(dict(stage="done",features=len(names),seconds=time.time()-started),flush=True)

