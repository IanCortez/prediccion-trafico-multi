from common import *
from numba import njit,prange

def node_connections(comp):
    n=len(comp['to_junction']);outj=np.full((n,2),-1,np.int32);inj=outj.copy();outl=np.full((n,2),np.inf);inl=outl.copy()
    for slot in range(2):
        chain=comp['node_chain'][:,slot];valid=chain>=0;pos=comp['node_pos'][valid,slot];c=chain[valid];outj[valid,slot]=comp['chain_dst'][c];inj[valid,slot]=comp['chain_src'][c];outl[valid,slot]=comp['chain_length'][c]-comp['prefix_length'][pos];inl[valid,slot]=comp['prefix_length'][pos]
    junction=comp['to_junction']>=0;outj[junction,0]=comp['to_junction'][junction];inj[junction,0]=comp['to_junction'][junction];outl[junction,0]=0;inl[junction,0]=0;outj[junction,1]=-1;inj[junction,1]=-1
    return outj,outl,inj,inl,comp['node_chain'],comp['node_pos'],comp['prefix_length']

@njit(cache=True)
def node_distance(a,b,D,outj,outl,inj,inl,chain,pos,prefix):
    if a==b:return 0.
    best=np.inf
    for s in range(2):
        source=outj[a,s]
        for t in range(2):
            target=inj[b,t]
            if source>=0 and target>=0:best=min(best,outl[a,s]+D[source,target]+inl[b,t])
            if chain[a,s]>=0 and chain[a,s]==chain[b,t] and pos[b,t]>=pos[a,s]:best=min(best,prefix[pos[b,t]]-prefix[pos[a,s]])
    return best

@njit(cache=True)
def edge_distance(ea,fa,eb,fb,u,v,length,D,outj,outl,inj,inl,chain,pos,prefix):
    remaining=(1-fa)*length[ea];tail=fb*length[eb]
    best=remaining+node_distance(v[ea],u[eb],D,outj,outl,inj,inl,chain,pos,prefix)+tail
    if ea==eb and fb>=fa:best=min(best,(fb-fa)*length[ea])
    if fa<=1e-7:best=min(best,node_distance(u[ea],u[eb],D,outj,outl,inj,inl,chain,pos,prefix)+tail)
    if fb>=1-1e-7:best=min(best,remaining+node_distance(v[ea],v[eb],D,outj,outl,inj,inl,chain,pos,prefix))
    if fa<=1e-7 and fb>=1-1e-7:best=min(best,node_distance(u[ea],v[eb],D,outj,outl,inj,inl,chain,pos,prefix))
    return best

@njit(cache=True,parallel=True)
def project_candidates(points,nearest,sample_edges,xy,u,v,k=4):
    n=len(points);edges=np.full((n,k),-1,np.int32);fraction=np.zeros((n,k),np.float32);error=np.full((n,k),np.inf,np.float32)
    for i in prange(n):
        for sample in nearest[i]:
            edge=sample_edges[sample];duplicate=False
            for j in range(k):
                if edges[i,j]==edge:duplicate=True;break
            if duplicate:continue
            a=xy[u[edge]];b=xy[v[edge]];delta=b-a;t=min(max(((points[i]-a)*delta).sum()/max((delta*delta).sum(),1e-12),0.),1.);residual=np.sqrt(((points[i]-a-t*delta)**2).sum())
            rank=k
            for j in range(k):
                if residual<error[i,j] or (residual==error[i,j] and edge<edges[i,j]):rank=j;break
            if rank<k:
                for j in range(k-1,rank,-1):edges[i,j]=edges[i,j-1];fraction[i,j]=fraction[i,j-1];error[i,j]=error[i,j-1]
                edges[i,rank]=edge;fraction[i,rank]=t;error[i,rank]=residual
        assert edges[i,0]>=0
        for j in range(1,k):
            if edges[i,j]<0:edges[i,j]=edges[i,0];fraction[i,j]=fraction[i,0];error[i,j]=error[i,0]
    return edges,fraction,error

@njit(cache=True,parallel=True)
def viterbi_match(points,offsets,edges,fraction,error,u,v,length,D,outj,outl,inj,inl,chain,pos,prefix,sigma=30.):
    n=len(points);best_edges=np.empty(n,np.int32);best_fraction=np.empty(n,np.float32);best_error=np.empty(n,np.float32);resets=np.zeros(len(offsets)-1,np.int32)
    for trip in prange(len(offsets)-1):
        start,end=offsets[trip],offsets[trip+1];L=end-start;k=edges.shape[1];back=np.zeros((L,k),np.int8);score=-.5*(error[start].astype(np.float64)/sigma)**2
        for t in range(1,L):
            i=start+t;gps=np.sqrt(((points[i]-points[i-1])**2).sum());previous_best=np.argmax(score);new=np.full(k,-np.inf)
            for current in range(k):
                if gps>500:
                    new[current]=-.5*(error[i,current]/sigma)**2;back[t,current]=previous_best;continue
                for previous in range(k):
                    distance=edge_distance(edges[i-1,previous],fraction[i-1,previous],edges[i,current],fraction[i,current],u,v,length,D,outj,outl,inj,inl,chain,pos,prefix)
                    if not np.isfinite(distance) or distance>750:continue
                    candidate=score[previous]-abs(distance-gps)/(50+.1*gps)-.5*(error[i,current]/sigma)**2
                    if candidate>new[current]:new[current]=candidate;back[t,current]=previous
            maximum=np.max(new)
            if not np.isfinite(maximum):
                resets[trip]+=1;new=-.5*(error[i].astype(np.float64)/sigma)**2
                for current in range(k):back[t,current]=previous_best
                maximum=np.max(new)
            elif gps>500:resets[trip]+=1
            score=new-maximum
        current=np.argmax(score)
        for t in range(L-1,-1,-1):
            i=start+t;best_edges[i]=edges[i,current];best_fraction[i]=fraction[i,current];best_error[i]=error[i,current]
            if t>0:current=back[t,current]
    return best_edges,best_fraction,best_error,resets
