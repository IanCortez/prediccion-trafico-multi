from common import *
from map_matching import edge_distance
from trazado_vial import collect_edge_path
from numba import njit,prange

@njit(cache=True)
def angle_bin(ax,ay,bx,by):
    return int(np.floor((np.arctan2(ax*by-ay*bx,ax*bx+ay*by)+np.pi/8)/(np.pi/4)))%8

@njit(cache=True,parallel=True)
def collect_turns(xy,offsets,fold,matched,fraction,error,u,v,length,D,conn,routing,arc_chain,to_junction,jxy,heading_in,heading_out,transition_offsets,chain_slot,destination_node):
    n_trans=transition_offsets[-1];local=np.zeros((5,8,4,n_trans),np.float32);global_turn=np.zeros((5,8,4,8),np.float32);counts=np.zeros((5,4),np.int64);samples=np.zeros((5,100000,7),np.int32);sample_n=np.zeros(5,np.int64)
    for f in prange(5):
        ids=np.empty(4096,np.int32);parts=np.empty(4096,np.float64)
        for trip in range(len(offsets)-1):
            if fold[trip]!=f:continue
            start,end=offsets[trip],offsets[trip+1];destination=xy[end-1]*1000;last_arc=-1;seen=False
            for point in range(start,end-1):
                gps=np.sqrt(((xy[point+1]-xy[point])**2).sum())*1000
                if gps>500 or error[point]>80 or error[point+1]>80:last_arc=-1;continue
                if gps<5:continue
                ea=matched[point];eb=matched[point+1];fa=fraction[point];fb=fraction[point+1];distance=edge_distance(ea,fa,eb,fb,u,v,length,D,*conn)
                if not np.isfinite(distance) or distance>750:last_arc=-1;continue
                if distance<5:continue
                used,_=collect_edge_path(ea,fa,eb,fb,u,v,length,D,conn,routing,ids,parts)
                if used<0:last_arc=-1;continue
                weight=max(.05,(1-max(error[point],error[point+1])/80)**2)
                for k in range(used):
                    edge=ids[k]
                    if last_arc>=0 and last_arc!=edge and v[last_arc]==u[edge]:
                        node=to_junction[u[edge]];incoming=arc_chain[last_arc];outgoing=arc_chain[edge]
                        if node>=0 and incoming!=outgoing:
                            dx=destination[0]-jxy[node,0];dy=destination[1]-jxy[node,1];remaining=np.sqrt(dx*dx+dy*dy);db=0 if remaining<500 else 1 if remaining<1500 else 2 if remaining<4000 else 3
                            direction=angle_bin(heading_in[incoming,0],heading_in[incoming,1],dx,dy);turn=angle_bin(heading_in[incoming,0],heading_in[incoming,1],heading_out[outgoing,0],heading_out[outgoing,1]);slot=chain_slot[outgoing];idx=transition_offsets[incoming]+slot
                            if idx>=transition_offsets[incoming+1]:counts[f,3]+=1;continue
                            local[f,direction,db,idx]+=weight;global_turn[f,direction,db,turn]+=weight;counts[f,0]+=1;seen=True
                            if (trip*1315423911+point*2654435761+edge*97)%1000==0:
                                sn=sample_n[f]
                                if sn<len(samples[f]):
                                    samples[f,sn,0]=incoming;samples[f,sn,1]=outgoing;samples[f,sn,2]=direction;samples[f,sn,3]=db;samples[f,sn,4]=destination_node[trip];samples[f,sn,5]=trip;samples[f,sn,6]=f;sample_n[f]+=1
                                else:counts[f,2]+=1
                    last_arc=edge
            counts[f,1]+=seen
    return local,global_turn,counts,samples,sample_n
