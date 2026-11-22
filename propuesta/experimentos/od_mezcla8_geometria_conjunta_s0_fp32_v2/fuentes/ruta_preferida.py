from common import *
from numba import njit
from preferencias_giro import angle_bin
from map_matching import node_distance
from trazado_vial import add_arc,add_chain,collect_node_path

@njit(cache=True)
def learned_path(base_ids,base_parts,base_used,eb,fb,destination,shortest,fold,u,v,length,D,conn,routing,to_junction,junction_raw,chain_dst,chain_length,prefix_length,outptr,outchains,transition_offsets,arc_chain,heading_in,heading_out,local,shared,xy,ids,parts,visited,stamp):
    used=0;current=-1;incoming=-1;travel=0.;decisions=0;support_sum=0.;nll=0.;fallback=0.;b=u[eb];budget=max(3*shortest,shortest+2000)
    for k in range(base_used):
        used=add_arc(base_ids[k],base_parts[k],ids,parts,used)
        if used<0:return -1,1.,0.,0.,0.
        travel+=base_parts[k];node=v[base_ids[k]];j=to_junction[node]
        if j>=0 and k<base_used-1:
            current=j;incoming=arc_chain[base_ids[k]];break
    if current<0:return used,0.,0.,0.,0.
    offsets,flat_arcs,_,_,_,_=routing
    while junction_raw[current]!=b and decisions<256:
        visited[current]=stamp;lo=outptr[current];hi=outptr[current+1];K=hi-lo
        if K==0:break
        dx=destination[0]-xy[junction_raw[current],0];dy=destination[1]-xy[junction_raw[current],1];remaining=np.sqrt(dx*dx+dy*dy);db=0 if remaining<500 else 1 if remaining<1500 else 2 if remaining<4000 else 3;ego=angle_bin(heading_in[incoming,0],heading_in[incoming,1],dx,dy)
        cost=np.full(K,np.inf);direct=np.full(K,-1,np.int64);turn=np.empty(K,np.int32);counts=np.zeros(K);global_counts=np.ones(8);multiplicity=np.zeros(8,np.int32)
        for f in range(5):
            if f!=fold:global_counts+=shared[f,ego,db]
        for s in range(K):
            c=outchains[lo+s];dest=chain_dst[c];end=offsets[c+1]-1
            if junction_raw[dest]==b:direct[s]=end
            else:
                for t in range(2):
                    if conn[4][b,t]==c:direct[s]=conn[5][b,t]
            if direct[s]>=0:
                cost[s]=prefix_length[direct[s]]+fb*length[eb]
            elif visited[dest]!=stamp:
                cost[s]=chain_length[c]+node_distance(junction_raw[dest],b,D,*conn)+fb*length[eb]
            if travel+cost[s]>budget:cost[s]=np.inf
            turn[s]=angle_bin(heading_in[incoming,0],heading_in[incoming,1],heading_out[c,0],heading_out[c,1]);multiplicity[turn[s]]+=1
            for f in range(5):
                if f!=fold:counts[s]+=local[f,ego,db,transition_offsets[incoming]+s]
        best=np.min(cost)
        if not np.isfinite(best):break
        prior=np.exp(-np.minimum((cost-best)/500,700))
        for s in range(K):prior[s]*=global_counts[turn[s]]/multiplicity[turn[s]]
        prior/=max(prior.sum(),1e-30);posterior=counts+20*prior
        for s in range(K):
            if not np.isfinite(cost[s]):posterior[s]=0.
        total=posterior.sum()
        if total<=0:break
        choice=np.argmax(posterior);c=outchains[lo+choice];decisions+=1;support_sum+=np.log1p(counts.sum());nll-=np.log(max(posterior[choice]/total,1e-12));end=direct[choice] if direct[choice]>=0 else offsets[c+1]-1
        used=add_chain(c,offsets[c],end,ids,parts,used,routing,length)
        if used<0:return -1,1.,0.,0.,decisions
        travel+=prefix_length[end]
        if direct[choice]>=0:
            used=add_arc(eb,fb*length[eb],ids,parts,used)
            return used,0.,support_sum/max(decisions,1),nll/max(decisions,1),decisions
        current=chain_dst[c];incoming=c
    if junction_raw[current]!=b:fallback=1.
    used=collect_node_path(junction_raw[current],b,D,conn,routing,length,ids,parts,used)
    if used>=0:used=add_arc(eb,fb*length[eb],ids,parts,used)
    return used,fallback,support_sum/max(decisions,1),nll/max(decisions,1),decisions
