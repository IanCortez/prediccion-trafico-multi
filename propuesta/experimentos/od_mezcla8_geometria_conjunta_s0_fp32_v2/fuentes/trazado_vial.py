from common import *
from map_matching import node_distance
from numba import njit

@njit(cache=True)
def choose_node_connection(a,b,D,conn):
    if a==b:return 0.,8
    outj,outl,inj,inl,chain,pos,prefix=conn;best=np.inf;which=-1
    for s in range(2):
        for t in range(2):
            source=outj[a,s];target=inj[b,t]
            if source>=0 and target>=0:
                cost=outl[a,s]+D[source,target]+inl[b,t]
                if cost<best:best=cost;which=s*2+t
            if chain[a,s]>=0 and chain[a,s]==chain[b,t] and pos[b,t]>=pos[a,s]:
                cost=prefix[pos[b,t]]-prefix[pos[a,s]]
                if cost<best:best=cost;which=4+s*2+t
    return best,which

@njit(cache=True)
def add_arc(edge,length,ids,parts,used):
    if length<=1e-8:return used
    if used>=len(ids):return -1
    ids[used]=edge;parts[used]=length;return used+1

@njit(cache=True)
def add_chain(chain,start,end,ids,parts,used,routing,rawlength):
    offsets,incoming,indptr,indices,chosen,chainlength=routing
    for p in range(start+1,end+1):
        edge=incoming[p];used=add_arc(edge,rawlength[edge],ids,parts,used)
        if used<0:return -1
    return used

@njit(cache=True)
def collect_node_path(a,b,D,conn,routing,rawlength,ids,parts,used):
    cost,which=choose_node_connection(a,b,D,conn)
    if which<0:return -1
    if which==8:return used
    offsets,incoming,indptr,indices,chosen,chainlength=routing;outj,outl,inj,inl,chain,pos,prefix=conn
    if which>=4:
        s=(which-4)//2;t=(which-4)%2;c=chain[a,s]
        return add_chain(c,pos[a,s],pos[b,t],ids,parts,used,routing,rawlength)
    s=which//2;t=which%2;c=chain[a,s]
    if c>=0:
        used=add_chain(c,pos[a,s],offsets[c+1]-1,ids,parts,used,routing,rawlength)
        if used<0:return -1
    current=outj[a,s];target=inj[b,t];steps=0
    while current!=target:
        best=np.inf;edge=-1
        for k in range(indptr[current],indptr[current+1]):
            candidate=chainlength[chosen[k]]+D[indices[k],target]
            if candidate<best:best=candidate;edge=k
        if edge<0:return -1
        c=chosen[edge];used=add_chain(c,offsets[c],offsets[c+1]-1,ids,parts,used,routing,rawlength)
        if used<0:return -1
        current=indices[edge];steps+=1
        if steps>len(D):return -1
    c=chain[b,t]
    if c>=0:used=add_chain(c,offsets[c],pos[b,t],ids,parts,used,routing,rawlength)
    return used

@njit(cache=True)
def collect_edge_path(ea,fa,eb,fb,u,v,length,D,conn,routing,ids,parts):
    outj,outl,inj,inl,chain,pos,prefix=conn;remaining=(1-fa)*length[ea];tail=fb*length[eb];a=v[ea];b=u[eb];first=remaining;last=tail;which=0
    best=remaining+node_distance(a,b,D,outj,outl,inj,inl,chain,pos,prefix)+tail
    if ea==eb and fb>=fa:
        cost=(fb-fa)*length[ea]
        if cost<best:best=cost;which=4
    if fa<=1e-7:
        cost=node_distance(u[ea],u[eb],D,outj,outl,inj,inl,chain,pos,prefix)+tail
        if cost<best:best=cost;which=1;a=u[ea];b=u[eb];first=0.;last=tail
    if fb>=1-1e-7:
        cost=remaining+node_distance(v[ea],v[eb],D,outj,outl,inj,inl,chain,pos,prefix)
        if cost<best:best=cost;which=2;a=v[ea];b=v[eb];first=remaining;last=0.
    if fa<=1e-7 and fb>=1-1e-7:
        cost=node_distance(u[ea],v[eb],D,outj,outl,inj,inl,chain,pos,prefix)
        if cost<best:best=cost;which=3;a=u[ea];b=v[eb];first=0.;last=0.
    if not np.isfinite(best):return -1,best
    if which==4:return add_arc(ea,best,ids,parts,0),best
    used=add_arc(ea,first,ids,parts,0)
    if used<0:return -1,best
    used=collect_node_path(a,b,D,conn,routing,length,ids,parts,used)
    if used<0:return -1,best
    used=add_arc(eb,last,ids,parts,used)
    return used,best
