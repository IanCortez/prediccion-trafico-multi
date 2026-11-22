from common import *
from numba import njit,prange
from map_matching import edge_distance
from ruteo_proyectado import collect_cost_path,path_shape
from ruta_preferida import learned_path
NAMES=['learned_shape_'+str(i) for i in range(14)]+['learned_'+s for s in ('polyline_km','circuity_log','turn_abs_pi','turn_signed_pi','sharp_turn_count','max_turn_pi','available','freeflow_600s','gps_time_600s','state_support','link_support','congestion_log','pace_log','fallback','mean_log_support','mean_selected_nll','junctions_per100')]

@njit(cache=True,parallel=True)
def learned_features(origin,destination,od_m,oe,of,od,de,df,dd,profile,u,v,length,free,xy,D,conn,routing,to_junction,junction_raw,chain_dst,chain_length,prefix_length,outptr,outchains,transition_offsets,arc_chain,heading_in,heading_out,local,shared,profile_time,state_support,link_support):
    n=len(origin);X=np.zeros((n,31),np.float32);errors=np.zeros((n,4),np.float32)
    for block in prange(64):
        base_ids=np.empty(4096,np.int32);base_parts=np.empty(4096);ids=np.empty(4096,np.int32);parts=np.empty(4096);vertices=np.empty((4100,2));cumulative=np.empty(4100);visited=np.zeros(len(junction_raw),np.int64)
        for q in range(block*n//64,(block+1)*n//64):
            best=np.inf;bo=-1;bd=-1
            for a in range(4):
                for b in range(4):
                    if od[q,a]>500 or dd[q,b]>500:continue
                    cost=edge_distance(oe[q,a],of[q,a],de[q,b],df[q,b],u,v,length,D,*conn);score=cost+3*(od[q,a]+dd[q,b])
                    if score<best:best=score;bo=a;bd=b
            if bo<0:
                X[q,14]=od_m[q]/1000;X[q,21]=od_m[q]/(25/3.6)/600;fallback=60+od_m[q]/(20/3.6);X[q,22]=fallback/600;X[q,25]=np.log(max(fallback/max(X[q,21]*600,1),.01));X[q,26]=np.log(max(1000*fallback/od_m[q],.01));X[q,27]=1.;continue
            ea=oe[q,bo];eb=de[q,bd];fa=of[q,bo];fb=df[q,bd];base_used,cost=collect_cost_path(ea,fa,eb,fb,u,v,length,length,D,conn,routing,base_ids,base_parts)
            if base_used<0:
                X[q,14]=od_m[q]/1000;X[q,21]=od_m[q]/(25/3.6)/600;X[q,22]=(60+od_m[q]/(20/3.6))/600;X[q,27]=1.;continue
            p=profile[q];fold=p//168
            used,fallback,support,nll,decisions=learned_path(base_ids,base_parts,base_used,eb,fb,destination[q],cost,fold,u,v,length,D,conn,routing,to_junction,junction_raw,chain_dst,chain_length,prefix_length,outptr,outchains,transition_offsets,arc_chain,heading_in,heading_out,local,shared,xy,ids,parts,visited,q+1)
            if used<0:
                used=base_used;ids[:used]=base_ids[:used];parts[:used]=base_parts[:used];fallback=1.;support=0.;nll=0.;decisions=0.
            gps=0.;ff=0.;state=0.;links=0.;total=0.
            for k in range(used):
                edge=ids[k];part=parts[k];ratio=part/length[edge];gps+=profile_time[p,edge]*ratio;ff+=free[edge]*ratio;state+=part*state_support[p,edge];links+=part*link_support[fold,edge];total+=part
                if k>0 and v[ids[k-1]]!=u[edge]:errors[q,0]+=1
                if part>length[edge]+.001:errors[q,1]=max(errors[q,1],part-length[edge])
            projected=xy[u[ea]]+(xy[v[ea]]-xy[u[ea]])*fa;distance=path_shape(origin[q],destination[q],projected,ids,parts,used,xy,u,v,length,vertices,cumulative,X[q]);end=xy[u[eb]]+(xy[v[eb]]-xy[u[eb]])*fb;snap=od[q,bo]+dd[q,bd]
            errors[q,2]=np.sqrt(((vertices[used+1]-end)**2).sum());errors[q,3]=abs(distance-total-snap);gps+=snap/(20/3.6);ff+=snap/(20/3.6)
            X[q,21]=ff/600;X[q,22]=gps/600;X[q,23]=state/max(total,1);X[q,24]=links/max(total,1);X[q,25]=np.log(max(gps/max(ff,1),.01));X[q,26]=np.log(max(1000*gps/max(distance,1),.01));X[q,27]=fallback;X[q,28]=support;X[q,29]=nll;X[q,30]=decisions/100
    return X,errors
