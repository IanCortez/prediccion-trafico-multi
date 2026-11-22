from common import *
from map_matching import edge_distance
from ruteo_proyectado import collect_cost_path,path_shape
from numba import njit,prange

@njit(cache=True,parallel=True)
def projected_route_features(origin,destination,od_m,oe,of,od,de,df,dd,profile,u,v,length,free,xy,arc_cost,D,conn,routing,profile_time,state_support,link_support,is_free):
    n=len(origin);static=np.zeros((n,26),np.float32);movement=np.zeros((n,5),np.float32);errors=np.zeros((n,3),np.float32)
    for block in prange(64):
        ids=np.empty(4096,np.int32);parts=np.empty(4096,np.float64);vertices=np.empty((4099,2),np.float64);cumulative=np.empty(4099,np.float64)
        for query in range(block*n//64,(block+1)*n//64):
            best=np.inf;second=np.inf;bo=-1;bd=-1;reachable=0
            for a in range(4):
                for b in range(4):
                    if od[query,a]>500 or dd[query,b]>500:continue
                    cost=edge_distance(oe[query,a],of[query,a],de[query,b],df[query,b],u,v,arc_cost,D,*conn)
                    if not np.isfinite(cost):continue
                    reachable+=1;snap=od[query,a]+dd[query,b];score=cost+snap/(10/3.6) if is_free else cost+3*snap
                    if score<best:second=best;best=score;bo=a;bd=b
                    elif score<second:second=score
            if bo<0:
                static[query,14]=od_m[query]/1000;static[query,21]=od_m[query]/(25/3.6)/600;static[query,22]=od[query,0]/1000;static[query,23]=dd[query,0]/1000
                fallback=60+od_m[query]/(20/3.6);movement[query,0]=fallback/600;movement[query,3]=np.log(max(fallback/(od_m[query]/(25/3.6)),.01));movement[query,4]=np.log(max(1000*fallback/od_m[query],.01));continue
            ea=oe[query,bo];eb=de[query,bd];fa=of[query,bo];fb=df[query,bd];used,cost=collect_cost_path(ea,fa,eb,fb,u,v,length,arc_cost,D,conn,routing,ids,parts)
            if used<0:
                static[query,14]=od_m[query]/1000;static[query,21]=od_m[query]/(25/3.6)/600;static[query,22]=od[query,0]/1000;static[query,23]=dd[query,0]/1000;fallback=60+od_m[query]/(20/3.6);movement[query,0]=fallback/600;movement[query,3]=np.log(max(fallback/(od_m[query]/(25/3.6)),.01));movement[query,4]=np.log(max(1000*fallback/od_m[query],.01));continue
            gps=0.;ff=0.;support=0.;links=0.;total_m=0.;planning=0.;p=profile[query];fold=p//168
            for k in range(used):
                edge=ids[k];part=parts[k];ratio=part/length[edge];gps+=profile_time[p,edge]*ratio;ff+=free[edge]*ratio;support+=part*state_support[p,edge];links+=part*link_support[fold,edge];total_m+=part;planning+=arc_cost[edge]*ratio
            errors[query,0]=abs(planning-cost);projected=xy[u[ea]]+(xy[v[ea]]-xy[u[ea]])*fa
            distance=path_shape(origin[query],destination[query],projected,ids,parts,used,xy,u,v,length,vertices,cumulative,static[query])
            end_projected=xy[u[eb]]+(xy[v[eb]]-xy[u[eb]])*fb
            errors[query,1]=np.sqrt(((vertices[used+1]-end_projected)**2).sum())
            snap=od[query,bo]+dd[query,bd];errors[query,2]=abs(distance-total_m-snap);gps+=snap/(20/3.6);ff+=snap/(20/3.6)
            static[query,21]=ff/600;static[query,22]=od[query,bo]/1000;static[query,23]=dd[query,bd]/1000;static[query,24]=(second-best)/(600 if is_free else 1000) if np.isfinite(second) else 0.;static[query,25]=reachable/16.
            movement[query,0]=gps/600;movement[query,1]=support/max(total_m,1);movement[query,2]=links/max(total_m,1);movement[query,3]=np.log(max(gps/max(ff,1),.01));movement[query,4]=np.log(max(1000*gps/max(distance,1),.01))
    return static,movement,errors

