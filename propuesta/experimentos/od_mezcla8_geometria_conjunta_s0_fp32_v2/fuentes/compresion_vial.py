from common import *
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def compress_graph(raw):
    n=len(raw['xy']);u,v=raw['u'],raw['v'];g=csr_matrix((np.ones(len(u)),(u,v)),shape=(n,n));sym=g+g.T;degree=np.diff(sym.indptr);ind=np.bincount(v,minlength=n);out=np.bincount(u,minlength=n)
    interior=(degree==2)&(((ind==2)&(out==2))|((ind==1)&(out==1)));junction=(degree>0)&~interior;_,component=connected_components(sym,directed=False)
    for cid in np.unique(component[degree>0]):
        ids=np.flatnonzero((component==cid)&(degree>0))
        if not junction[ids].any():junction[ids[0]]=True;interior[ids[0]]=False
    jraw=np.flatnonzero(junction);to_j=np.full(n,-1,np.int32);to_j[jraw]=np.arange(len(jraw));adj=[[] for _ in range(n)]
    for e,(a,b) in enumerate(zip(u,v)):adj[int(a)].append((int(b),e))
    src=[];dst=[];offsets=[0];nodes=[];incoming=[];length=[];free=[];pl=[];pf=[];node_chain=np.full((n,2),-1,np.int32);node_pos=node_chain.copy();slot=np.zeros(n,np.int8)
    for source in jraw:
        for neighbor,edge in adj[source]:
            previous=int(source);current=neighbor;ns=[int(source),current];es=[edge]
            while not junction[current]:
                options=[(w,e) for w,e in adj[current] if w!=previous];assert len(options)==1
                following,e=options[0];previous,current=current,following;ns.append(current);es.append(e);assert len(ns)<=n+1
            cid=len(src);src.append(to_j[source]);dst.append(to_j[current]);start=len(nodes);nodes.extend(ns);incoming.extend([-1]+es);plen=np.r_[0,np.cumsum(raw['length'][es])];ptime=np.r_[0,np.cumsum(raw['seconds'][es])];pl.extend(plen);pf.extend(ptime);length.append(plen[-1]);free.append(ptime[-1])
            for p,node in enumerate(ns[1:-1],1):
                s=slot[node];assert s<2;node_chain[node,s]=cid;node_pos[node,s]=start+p;slot[node]+=1
            offsets.append(len(nodes))
    incoming=np.array(incoming,np.int32);visited=np.bincount(incoming[incoming>=0],minlength=len(u));assert np.all(visited==1)
    return dict(junction_raw=jraw,to_junction=to_j,chain_src=np.array(src,np.int32),chain_dst=np.array(dst,np.int32),chain_offsets=np.array(offsets,np.int64),flat_nodes=np.array(nodes,np.int32),flat_incoming_arc=incoming,node_chain=node_chain,node_pos=node_pos,prefix_length=np.array(pl),prefix_free=np.array(pf),chain_length=np.array(length),chain_free=np.array(free))
