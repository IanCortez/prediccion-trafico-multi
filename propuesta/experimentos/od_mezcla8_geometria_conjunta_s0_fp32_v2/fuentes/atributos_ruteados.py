from common import *
from numba import njit,prange
from map_matching import edge_distance
from ruteo_proyectado import collect_cost_path
FIELDS=['intersections','signals','stop','give_way','crossings','bridge_fraction','tunnel_fraction','roundabout_fraction','lanes_known_fraction','lanes_mean_known','road_class_changes','speed_limit_std_kmh']+['class_fraction_'+str(i) for i in range(15)]+['available','intersections_per_km','signals_per_km']

@njit(cache=True,parallel=True)
def attribute_features(oe,of,od,de,df,dd,u,v,length,free,arc_cost,D,conn,routing,road_class,lanes,flags,node_control,degree,is_free):
    n=len(oe);X=np.zeros((n,30),np.float32);checks=np.zeros(n,np.float32)
    for block in prange(64):
        ids=np.empty(4096,np.int32);parts=np.empty(4096)
        for q in range(block*n//64,(block+1)*n//64):
            best=np.inf;bo=-1;bd=-1
            for a in range(4):
                for b in range(4):
                    if od[q,a]>500 or dd[q,b]>500:continue
                    cost=edge_distance(oe[q,a],of[q,a],de[q,b],df[q,b],u,v,arc_cost,D,*conn);snap=od[q,a]+dd[q,b];score=cost+snap/(10/3.6) if is_free else cost+3*snap
                    if score<best:best=score;bo=a;bd=b
            if bo<0:continue
            used,cost=collect_cost_path(oe[q,bo],of[q,bo],de[q,bd],df[q,bd],u,v,length,arc_cost,D,conn,routing,ids,parts)
            if used<=0:continue
            total=0.;known=0.;lanes_sum=0.;speed_sum=0.;speed2_sum=0.;planning=0.
            for k in range(used):
                e=ids[k];m=parts[k];total+=m;planning+=m/length[e]*arc_cost[e];X[q,12+road_class[e]]+=m
                for f in range(3):X[q,5+f]+=m*flags[e,f]
                if lanes[e]>0:known+=m;lanes_sum+=m*lanes[e]
                speed=3.6*length[e]/free[e];speed_sum+=m*speed;speed2_sum+=m*speed*speed
                if k and road_class[e]!=road_class[ids[k-1]]:X[q,10]+=1
                if k<used-1 or m>=length[e]-1e-4:
                    node=v[e];X[q,0]+=degree[node]>=3
                    for c in range(4):X[q,c+1]+=node_control[node,c]
            X[q,11]=np.sqrt(max(speed2_sum/max(total,1e-8)-(speed_sum/max(total,1e-8))**2,0.))
            checks[q]=abs(planning-cost);X[q,5:8]/=max(total,1e-8);X[q,8]=known/max(total,1e-8);X[q,9]=lanes_sum/max(known,1e-8);X[q,12:27]/=max(total,1e-8);X[q,27]=1.;X[q,28]=X[q,0]/max(total/1000,.001);X[q,29]=X[q,1]/max(total/1000,.001)
    return X,checks
