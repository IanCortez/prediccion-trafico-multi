from common import *
from features import ORIGIN
import pandas as pd
from numba import njit,prange,set_num_threads
@njit(cache=True,parallel=True)
def accumulate(xy,offsets,local,fold):
    fd=np.zeros((5,12,90000,8),np.float64);ft=np.zeros_like(fd);fs=np.zeros((5,12,90000),np.float64)
    cd=np.zeros((5,12,10000,8),np.float64);ct=np.zeros_like(cd);cs=np.zeros((5,12,10000),np.float64)
    counts=np.zeros((5,4),np.int64)
    for f in prange(5):
        for i in range(len(local)):
            if fold[i]!=f:continue
            start=offsets[i];end=offsets[i+1]
            for j in range(start,end-1):
                dx=(xy[j+1,0]-xy[j,0])*1000;dy=(xy[j+1,1]-xy[j,1])*1000
                distance=np.sqrt(dx*dx+dy*dy)
                if distance>500:counts[f,2]+=1;continue
                x=(xy[j+1,0]+xy[j,0])*500;y=(xy[j+1,1]+xy[j,1])*500
                ix=int(np.floor((x+15000)/100));iy=int(np.floor((y+15000)/100))
                if ix<0 or ix>=300 or iy<0 or iy>=300:counts[f,3]+=1;continue
                cell=ix*300+iy;coarse=(ix//3)*100+(iy//3)
                clock=local[i]+(j-start)*15+7
                dow=(clock//86400+3)%7;hour=(clock%86400)//3600
                state=(1 if dow>=5 else 0)*6+hour//4
                if distance<5:
                    fs[f,state,cell]+=15;cs[f,state,coarse]+=15;counts[f,1]+=1
                else:
                    direction=int(np.floor((np.arctan2(dy,dx)+np.pi/8)/(np.pi/4)))%8
                    fd[f,state,cell,direction]+=distance;ft[f,state,cell,direction]+=15
                    cd[f,state,coarse,direction]+=distance;ct[f,state,coarse,direction]+=15;counts[f,0]+=1
    return fd,ft,fs,cd,ct,cs,counts
started=time.time();set_num_threads(4)
z=np.load(DATA/"trayectorias_train.npz");ts=z["timestamp"]
local=pd.to_datetime(ts,unit="s",utc=True).tz_convert("Europe/Lisbon").tz_localize(None).as_unit("s").asi8
fold=(ts//(7*86400))%5
fd,ft,fs,cd,ct,cs,counts=accumulate(z["xy"],z["offsets"],local,fold)
print(dict(stage="aggregated",seconds=time.time()-started,counts=counts.tolist()),flush=True)
np.savez_compressed(DATA/"movimiento_gps_bloques.npz",fine_distance=fd.astype(np.float32),fine_time=ft.astype(np.float32),fine_stop=fs.astype(np.float32),coarse_distance=cd.astype(np.float32),coarse_time=ct.astype(np.float32),coarse_stop=cs.astype(np.float32),counts=counts)
raw=np.load(DATA/"mapa_dirigido.npz");a=raw["xy"][raw["u"]];b=raw["xy"][raw["v"]]
mid=(a+b)/2;ii=np.floor((mid+15000)/100).astype(int)
valid=(ii>=0).all(1)&(ii<300).all(1)
fine=np.clip(ii[:,0],0,299)*300+np.clip(ii[:,1],0,299)
coarse=np.clip(ii[:,0]//3,0,99)*100+np.clip(ii[:,1]//3,0,99)
direction=(np.floor((np.arctan2(b[:,1]-a[:,1],b[:,0]-a[:,0])+np.pi/8)/(np.pi/4)).astype(int))%8
prior=2*raw["seconds"]/raw["length"]
full=[v.sum(0) for v in (fd,ft,fs,cd,ct,cs)]
times=np.empty((72,len(a)),np.float32);support=np.empty_like(times)
for f in range(6):
    D,T,S,C,U,V=[total-arr[f] if f<5 else total for total,arr in zip(full,(fd,ft,fs,cd,ct,cs))]
    sumT=T.sum(-1);sumU=U.sum(-1)
    for st in range(12):
        dt=D[st,fine,direction];tt=T[st,fine,direction];stop=S[st,fine]*(tt+15)/(sumT[st,fine]+120)
        dc=C[st,coarse,direction];tc=U[st,coarse,direction];stopc=V[st,coarse]*(tc+15)/(sumU[st,coarse]+120)
        parent=(tc+stopc+2500*prior)/(dc+2500)
        pace=(tt+stop+500*parent)/(dt+500)
        pace=np.where(valid,pace,prior)
        pace=np.clip(pace,3.6/120,3.6/2)
        times[f*12+st]=raw["length"]*pace
        confidence=1-(500/(dt+500))*(2500/(dc+2500))
        support[f*12+st]=raw["length"]*np.where(valid,confidence,0)
np.savez_compressed(DATA/"arcos_tiempos_gps.npz",time_s=times,observed_length_m=support)
write_json(DATA/"movimiento_gps_manifiesto.json",dict(trips=len(ts),gps_points=len(z["xy"]),counts_by_fold=counts,source_sha256=sha(DATA/"trayectorias_train.npz"),edge_profiles_sha256=sha(DATA/"arcos_tiempos_gps.npz"),states="6 departure/mid-interval local four-hour blocks x weekday/weekend",directions=8,fine_cell_m=100,coarse_cell_m=300,folds="UTC departure week modulo 5; all intervals of query-fold trips excluded",out_of_fold_profiles=5,full_profile_index=5,assumed_interval_s=15,stopped_threshold_m_per_15s=5,rejected_jump_m=500,shrinkage_distance_m=[500,2500],fallback="OSM free-flow pace multiplied by 2",speed_clip_kmh=[2,120],limitation="Local GPS interval pace assigned to cells/directions; not claimed as exact map-matched link speed or live traffic",test_labels_read=False,seconds=time.time()-started))
print(dict(stage="done",shape=times.shape,seconds=time.time()-started),flush=True)

