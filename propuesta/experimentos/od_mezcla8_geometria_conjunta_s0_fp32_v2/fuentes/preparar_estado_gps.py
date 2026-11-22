from common import *
from features import ORIGIN
import torch,sys,pandas as pd
from numba import njit,prange,set_num_threads
sys.path.insert(0,str(ROOT/"codigo-ian/src"));import tte_model as M
@njit(cache=True,parallel=True)
def summarize(xy,offsets,ts,fold,transitions,utc_offsets):
    dist=np.zeros((2,168,253),np.float64);seconds=np.zeros_like(dist);counts=np.zeros((2,3),np.int64)
    dlat=500/110540;dlon=500/(111320*np.cos(np.deg2rad((41.1405+41.1865)/2)))
    for f in prange(2):
        for i in range(len(ts)):
            if fold[i]!=f:continue
            for k in range(offsets[i],offsets[i+1]-1):
                dx=(xy[k+1,0]-xy[k,0])*1000;dy=(xy[k+1,1]-xy[k,1])*1000
                length=np.sqrt(dx*dx+dy*dy)
                if length>500:counts[f,1]+=1;continue
                xm=(xy[k+1,0]+xy[k,0])/2;ym=(xy[k+1,1]+xy[k,1])/2
                lon=-8.62+xm/(111.195*np.cos(np.deg2rad(41.16)));lat=41.16+ym/111.195
                if not (41.1405<=lat<=41.1865 and -8.6887<=lon<=-8.5557):counts[f,2]+=1;continue
                cell=int((lat-41.1405)//dlat)*23+int((lon+8.6887)//dlon)
                instant=ts[i]+(k-offsets[i])*15+7
                segment=np.searchsorted(transitions,instant,side="right")-1;wall=instant+utc_offsets[segment]
                hour=(wall%86400)//3600;dow=(wall//86400+3)%7;how=dow*24+hour
                dist[f,how,cell]+=0 if length<5 else length;seconds[f,how,cell]+=15;counts[f,0]+=1
    return dist,seconds,counts
started=time.time();set_num_threads(2)
cd=torch.load(DATA/"original_completo_cd_v1.pt",map_location="cpu",weights_only=False)
d=np.load(DATA/"train_x.npz");z=np.load(DATA/"trayectorias_train.npz")
order=np.argsort(d["row"]);pos=np.searchsorted(d["row"][order],z["row"]);ix=order[pos]
assert np.array_equal(d["row"][ix],z["row"])
fold=cd.st.fold.numpy()[ix]
ts=z["timestamp"];low=pd.Timestamp(int(ts.min()),unit="s",tz="UTC").floor("D");high=pd.Timestamp(int(ts.max())+1800,unit="s",tz="UTC").ceil("D")
hours=pd.date_range(low,high,freq="h");utc=hours.as_unit("s").asi8;wall=hours.tz_convert("Europe/Lisbon").tz_localize(None).as_unit("s").asi8
off=wall-utc;starts=np.r_[0,np.flatnonzero(np.diff(off)!=0)+1];transitions=utc[starts];utc_offsets=off[starts]
dist,seconds,counts=summarize(z["xy"],z["offsets"],ts,fold,transitions,utc_offsets)
full_d=dist.sum(0);full_s=seconds.sum(0);speed=[];obs=[]
for f in range(3):
    D=full_d-dist[f] if f<2 else full_d;S=full_s-seconds[f] if f<2 else full_s
    v=(D+500)/(S+120);v=np.clip(v,.1,120/3.6)
    norm=(np.log(v)-np.log(15/3.6))/np.log(2)
    known=S>0;speed.append(np.where(known,norm,0).astype(np.float32));obs.append(known.astype(np.float32))
cd.st.spd=torch.tensor(np.stack(speed));cd.st.obs=torch.tensor(np.stack(obs))
torch.save(cd,DATA/"original_completo_gps_state_v1.pt")
write_json(DATA/"estado_gps_real_manifiesto.json",dict(fixed_grid_cells=253,speed_units="log(actual GPS neighborhood speed m/s / (15 km/h)) / log(2)",pseudo_distance_m=500,pseudo_seconds=120,source_rows=len(ts),original_training_rows=len(d["row"]),fold_match=True,counts=counts,observed_fraction=np.stack(obs).mean(axis=(1,2)),timezone_transitions=transitions,timezone_offsets=utc_offsets,query_fold_excluded=True,train_only=True,original_cache_sha256=sha(DATA/"original_completo_cd_v1.pt"),new_cache_sha256=sha(DATA/"original_completo_gps_state_v1.pt"),seconds=time.time()-started))
print(dict(stage="done",counts=counts.tolist(),coverage=np.stack(obs).mean((1,2)).tolist(),seconds=time.time()-started),flush=True)


