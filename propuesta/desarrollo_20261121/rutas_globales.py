"""Screen global historical-cost routing against geometry; observed routes are targets only."""
from piloto_transferencia import ROOT,BASE,OUT,write,sha,paired,metric
import numpy as np,pandas as pd,time,json
from scipy.spatial import cKDTree
from numba import njit
import heapq

@njit(cache=False)
def astar(start,end,xy,ptr,dest,edgeids,length,free,times,support,profile,kind):
 n=len(xy);g=np.full(n,np.inf);parent=np.full(n,-1,np.int32);parent_edge=np.full(n,-1,np.int32);done=np.zeros(n,np.bool_);g[start]=0.
 hscale=1. if kind==0 else 0.03 if kind==1 else 0.012
 h=np.sqrt(((xy[start]-xy[end])**2).sum())*hscale;heap=[(h,0.,start)];expanded=0
 while heap:
  _,score,node=heapq.heappop(heap)
  if done[node]:continue
  done[node]=True;expanded+=1
  if node==end:break
  if expanded>40000:return np.empty(0,np.int32),np.empty(0,np.int32),expanded
  for q in range(ptr[node],ptr[node+1]):
   e=edgeids[q];v=dest[q]
   cost=length[e] if kind==0 else times[profile,e] if kind==1 else times[profile,e]*(1-.6*support[profile//168,e])
   assert cost>0
   alt=score+cost
   if alt<g[v]:
    g[v]=alt;parent[v]=node;parent_edge[v]=e;heuristic=np.sqrt(((xy[v]-xy[end])**2).sum())*hscale;heapq.heappush(heap,(alt+heuristic,alt,np.int64(v)))
 if not done[end]:return np.empty(0,np.int32),np.empty(0,np.int32),expanded
 nodes=np.empty(4096,np.int32);edges=np.empty(4096,np.int32);used=0;at=end
 while at!=start and used<4095:
  nodes[used]=at;edges[used]=parent_edge[at];at=parent[at];used+=1
 if at!=start:return np.empty(0,np.int32),np.empty(0,np.int32),expanded
 nodes[used]=start
 return nodes[:used+1][::-1],edges[:used][::-1],expanded

def shape(points):
 segment=np.sqrt((np.diff(points,axis=0)**2).sum(1));s=np.r_[0.,np.cumsum(segment)];total=s[-1];q=np.arange(1,8)/8
 target=total*q;out=np.column_stack([np.interp(target,s,points[:,j]) for j in (0,1)]);chord=points[0]*(1-q[:,None])+points[-1]*q[:,None]
 return ((out-chord)/1000).reshape(-1),total/1000

def main():
 start=time.time();run=OUT/'rutas_globales';run.mkdir(exist_ok=True);assert not (run/'resultado.json').exists()
 protocol=dict(kind='Global A-star with historical expected edge times; geometry targets only in screening',variants=['shortest_length_control','historical_time','historical_time_support_preference'],preference='cost = historical time * (1 - 0.6 link_support); positive and globally optimized',training_profiles='Exclude query UTC departure-week fold',validation_profiles='Training history only, index5',endpoint_attachment='nearest raw road node; same for three variants, recorded limitation',training_n=2048,validation_n=2048,seed=4321,expansion_gate='Held-out geometry mean error at least 5% lower than same-endpoint shortest control, with paired day CI excluding zero; route variant chosen using Train only',Test_access=False)
 write(run/'protocolo.json',protocol)
 raw=dict(np.load(BASE/'datos/mapa_dirigido_2014.npz'));xy=raw['xy'];order=np.argsort(raw['u'],kind='stable').astype(np.int32);ptr=np.r_[0,np.cumsum(np.bincount(raw['u'],minlength=len(xy)))].astype(np.int64);dest=raw['v'][order];tree=cKDTree(xy);times=np.load(BASE/'datos/perfiles_mapmatched_2014/time_s.npy',mmap_mode='r');support=np.load(BASE/'datos/perfiles_mapmatched_2014/link_support.npy',mmap_mode='r');rg=np.random.default_rng(4321);results={}
 for split in ['train','validation']:
  d=dict(np.load(BASE/'datos'/f'{split}_x.npz'));geo=dict(np.load(BASE/'datos'/f'geometria_{split}.npz'));gs=np.argsort(geo['row']);geo['row']=geo['row'][gs];geo['X']=geo['X'][gs];assert np.all(np.diff(geo['row'])>0);ix=np.sort(rg.choice(len(d['row']),2048,replace=False));selected={k:a[ix] for k,a in d.items()};pos=np.searchsorted(geo['row'],selected['row']);safe=np.minimum(pos,len(geo['row'])-1);valid=(pos<len(geo['row']))&(geo['row'][safe]==selected['row'])&(geo['X'][safe,-1]<=500);gold=geo['X'][safe,1:15];valid&=np.isfinite(gold).all(1)
  origin=np.column_stack([(selected['o_lon']+8.62)*111195*np.cos(np.deg2rad(41.16)),(selected['o_lat']-41.16)*111195]);end=np.column_stack([(selected['d_lon']+8.62)*111195*np.cos(np.deg2rad(41.16)),(selected['d_lat']-41.16)*111195]);eo,src=tree.query(origin);ed,tar=tree.query(end);dt=pd.to_datetime(selected['timestamp'],unit='s',utc=True).tz_convert('Europe/Lisbon');hour=np.asarray(dt.dayofweek)*24+np.asarray(dt.hour);fold=(selected['timestamp']//(7*86400))%5 if split=='train' else np.full(len(ix),5);profiles=(fold*168+hour).astype(np.int32);shapes=np.zeros((len(ix),3,14),np.float32);stats=np.zeros((len(ix),3,6),np.float32);ok=np.ones((len(ix),3),bool)
  for i in range(len(ix)):
   for kind in range(3):
    nodes,edges,expanded=astar(int(src[i]),int(tar[i]),xy,ptr,dest,order,raw['length'],raw['seconds'],times,support,int(profiles[i]),kind)
    if not len(nodes):ok[i,kind]=False;continue
    points=np.vstack([origin[i],xy[nodes],end[i]]);shapes[i,kind],km=shape(points);stats[i,kind]=[km,raw['seconds'][edges].sum()/600,times[profiles[i],edges].sum()/600,support[fold[i],edges].mean() if len(edges) else 0,expanded,(eo[i]+ed[i])/1000]
   if (i+1)%512==0:print(json.dumps(dict(split=split,done=i+1,seconds=time.time()-start)),flush=True)
  assert valid.mean()>.7;mask=valid&ok.all(1);error=np.abs(shapes-gold[:,None,:]).mean(2);results[split]=dict(row=selected['row'],timestamp=selected['timestamp'],mask=mask,error=error,shapes=shapes,stats=stats,ok=ok,indices=ix)
  np.savez_compressed(run/f'{split}_diagnostico.npz',**results[split]);print(split,'errors',error[mask].mean(0),flush=True)
 choice=int(np.argmin(results['train']['error'][results['train']['mask']].mean(0)))
 a=results['validation'];m=a['mask'];delta=paired(np.zeros(m.sum()),a['error'][m,choice]*1000,a['error'][m,0]*1000,a['timestamp'][m]);relative=1-a['error'][m,choice].mean()/a['error'][m,0].mean();passed=choice!=0 and relative>=.05 and delta['ci95_s'][1]<0
 result=dict(protocol=protocol,train_geometry_coordinate_mae_m=(results['train']['error'][results['train']['mask']].mean(0)*1000),validation_geometry_coordinate_mae_m=a['error'][m].mean(0)*1000,validation_n=int(m.sum()),selected_kind_using_train=choice,relative_improvement=float(relative),paired_geometry_delta_m=delta,passed_expansion_gate=bool(passed),query_geometry_is_input=False,Test_access=False,seconds=time.time()-start,code_sha256=sha(ROOT.__class__(__file__)),conclusion='Geometry diagnostic is not duration MAE. Expand only if global routes beat control; no claim of full decision-focused neural training.')
 write(run/'resultado.json',result);print(json.dumps(result,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else str(v)),flush=True)
if __name__=='__main__':main()


