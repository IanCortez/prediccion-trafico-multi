from common import *
from numba import njit,prange
@njit(cache=True,parallel=True)
def chain_profiles(offsets,incoming,time_arc,support_arc):
    n=len(offsets)-1;profiles=time_arc.shape[1]
    pt=np.zeros((len(incoming),profiles),np.float32);pc=np.zeros_like(pt)
    totals=np.zeros((n,profiles),np.float32);coverage=np.zeros_like(totals)
    for c in prange(n):
        start,end=offsets[c],offsets[c+1]
        for node in range(start+1,end):
            arc=incoming[node]
            for p in range(profiles):
                pt[node,p]=pt[node-1,p]+time_arc[arc,p]
                pc[node,p]=pc[node-1,p]+support_arc[arc,p]
        totals[c]=pt[end-1];coverage[c]=pc[end-1]
    return pt,pc,totals,coverage
@njit(cache=True,parallel=True)
def routed_profiles(pred,sources,indptr,indices,chosen,chain_time,chain_support,chain_length,chain_free,query_ids,bounds,destination,profile,n_queries):
    out=np.full((n_queries,4),np.nan,np.float32)
    nj=pred.shape[1];npf=chain_time.shape[1]
    for source_row in prange(len(sources)):
        lo,hi=bounds[source_row],bounds[source_row+1]
        if lo==hi:continue
        mt=np.full((npf,nj),np.nan,np.float32);mc=np.full_like(mt,np.nan)
        ml=np.full(nj,np.nan,np.float64);mf=np.full_like(ml,np.nan)
        root=sources[source_row];mt[:,root]=0;mc[:,root]=0;ml[root]=0;mf[root]=0
        stack=np.empty(nj,np.int32)
        for qi in range(lo,hi):
            q=query_ids[qi];p=profile[q];dest=destination[q]
            if np.isnan(mt[p,dest]):
                node=dest;size=0
                while node>=0 and np.isnan(mt[p,node]) and size<nj:
                    stack[size]=node;size+=1;node=pred[source_row,node]
                if node<0 or size>=nj:continue
                while size>0:
                    size-=1;node=stack[size];parent=pred[source_row,node]
                    edge=-1
                    for k in range(indptr[parent],indptr[parent+1]):
                        if indices[k]==node:edge=k;break
                    if edge<0:break
                    c=chosen[edge]
                    mt[p,node]=mt[p,parent]+chain_time[c,p]
                    mc[p,node]=mc[p,parent]+chain_support[c,p]
                    if np.isnan(ml[node]):
                        ml[node]=ml[parent]+chain_length[c];mf[node]=mf[parent]+chain_free[c]
            out[q,0]=mt[p,dest];out[q,1]=mc[p,dest];out[q,2]=ml[dest];out[q,3]=mf[dest]
    return out

