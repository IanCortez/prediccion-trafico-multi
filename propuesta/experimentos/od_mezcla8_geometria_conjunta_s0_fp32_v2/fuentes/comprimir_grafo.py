from common import *
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
started=time.time()
raw=np.load(DATA/"mapa_dirigido.npz");n=len(raw["xy"]);u,v=raw["u"],raw["v"]
g=csr_matrix((np.ones(len(u)),(u,v)),shape=(n,n));sym=g+g.T;degree=np.diff(sym.indptr)
ind=np.bincount(v,minlength=n);out=np.bincount(u,minlength=n)
interior=(degree==2)&(((ind==2)&(out==2))|((ind==1)&(out==1)))
junction=(degree>0)&~interior
_,component=connected_components(sym,directed=False)
for comp in np.unique(component[degree>0]):
    ids=np.flatnonzero((component==comp)&(degree>0))
    if not junction[ids].any():junction[ids[0]]=True;interior[ids[0]]=False
jraw=np.flatnonzero(junction);to_j=np.full(n,-1,dtype=np.int32);to_j[jraw]=np.arange(len(jraw))
adj=[[] for _ in range(n)]
for e,(a,b) in enumerate(zip(u,v)):adj[int(a)].append((int(b),e))
src=[];dst=[];chain_offsets=[0];nodes=[];incoming=[];lengths=[];free=[];prefix_len=[];prefix_free=[]
node_chain=np.full((n,2),-1,dtype=np.int32);node_pos=node_chain.copy();slot=np.zeros(n,dtype=np.int8)
for source in jraw:
    for neighbor,edge in adj[source]:
        previous=int(source);current=neighbor;ns=[int(source),current];es=[edge]
        while not junction[current]:
            options=[(w,e) for w,e in adj[current] if w!=previous]
            assert len(options)==1,(current,options)
            following,e=options[0];previous,current=current,following;ns.append(current);es.append(e)
            assert len(ns)<=n+1,"Unclosed chain"
        cid=len(src);src.append(int(to_j[source]));dst.append(int(to_j[current]))
        start=len(nodes);nodes.extend(ns);incoming.extend([-1]+es)
        plen=np.r_[0,np.cumsum(raw["length"][es])];ptime=np.r_[0,np.cumsum(raw["seconds"][es])]
        prefix_len.extend(plen);prefix_free.extend(ptime);lengths.append(plen[-1]);free.append(ptime[-1])
        for p,node in enumerate(ns[1:-1],1):
            s=slot[node];assert s<2,(node,s)
            node_chain[node,s]=cid;node_pos[node,s]=start+p;slot[node]+=1
        chain_offsets.append(len(nodes))
np.savez_compressed(DATA/"grafo_compacto.npz",junction_raw=jraw,to_junction=to_j,chain_src=np.array(src,dtype=np.int32),chain_dst=np.array(dst,dtype=np.int32),chain_offsets=np.array(chain_offsets,dtype=np.int64),flat_nodes=np.array(nodes,dtype=np.int32),flat_incoming_arc=np.array(incoming,dtype=np.int32),node_chain=node_chain,node_pos=node_pos,prefix_length=np.array(prefix_len),prefix_free=np.array(prefix_free),chain_length=np.array(lengths),chain_free=np.array(free))
visited=np.bincount(np.array(incoming)[np.array(incoming)>=0],minlength=len(u))
assert (visited==1).all(),dict(missing=int((visited==0).sum()),repeated=int((visited>1).sum()))
write_json(DATA/"grafo_compacto_manifiesto.json",dict(raw_nodes=n,junctions=len(jraw),directed_chains=len(src),raw_arcs=len(u),all_raw_arcs_covered_exactly_once=True,method="Exact contraction of homogeneous directed/bidirectional degree-2 chains; retain cumulative original geometry and direction",raw_map_sha256=sha(DATA/"mapa_dirigido.npz"),compact_sha256=sha(DATA/"grafo_compacto.npz"),seconds=time.time()-started))
print(dict(junctions=len(jraw),chains=len(src),seconds=time.time()-started),flush=True)

