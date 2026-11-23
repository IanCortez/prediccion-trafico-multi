from base_replanteo import np
from numba import njit,prange
from map_matching import edge_distance
@njit(cache=True,parallel=True)
def viterbi_trace(points,offsets,edges,fraction,error,u,v,length,D,outj,outl,inj,inl,chain,pos,prefix,sigma=30.):
    n=len(points);reset_flags=np.zeros(n,np.int8);best_edges=np.empty(n,np.int32);best_fraction=np.empty(n,np.float32);best_error=np.empty(n,np.float32);resets=np.zeros(len(offsets)-1,np.int32)
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
                reset_flags[i]=2;resets[trip]+=1;new=-.5*(error[i].astype(np.float64)/sigma)**2
                for current in range(k):back[t,current]=previous_best
                maximum=np.max(new)
            elif gps>500:resets[trip]+=1;reset_flags[i]=1
            score=new-maximum
        current=np.argmax(score)
        for t in range(L-1,-1,-1):
            i=start+t;best_edges[i]=edges[i,current];best_fraction[i]=fraction[i,current];best_error[i]=error[i,current]
            if t>0:current=back[t,current]
    return best_edges,best_fraction,best_error,resets,reset_flags
