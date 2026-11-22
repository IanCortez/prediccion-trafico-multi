"""Generate one global historical route per query. No dense distance cache, no Test."""
from rutas_globales import astar
from piloto_transferencia import ROOT,BASE,OUT,write,sha
from numba import njit,prange,set_num_threads
from scipy.spatial import cKDTree
import numpy as np,pandas as pd,time,json

@njit(parallel=True,cache=False)
def route_batch(origin,end,src,tar,profiles,xy,ptr,dest,edgeids,length,free,times,support):
 out=np.zeros((len(src),21),np.float32)
 for i in prange(len(src)):
  nodes,edges,expanded=astar(src[i],tar[i],xy,ptr,dest,edgeids,length,free,times,support,profiles[i],2)
  connector=np.sqrt(((origin[i]-xy[src[i]])**2).sum())+np.sqrt(((end[i]-xy[tar[i]])**2).sum())
  if len(nodes)==0 or connector>500:
   od=np.sqrt(((origin[i]-end[i])**2).sum());out[i,14]=od/1000;out[i,15]=od/(25/3.6)/600;out[i,16]=out[i,15];out[i,19]=connector/1000;continue
  points=np.empty((len(nodes)+2,2));points[0]=origin[i];points[-1]=end[i]
  for j in range(len(nodes)):points[j+1]=xy[nodes[j]]
  s=np.zeros(len(points))
  for j in range(1,len(points)):s[j]=s[j-1]+np.sqrt(((points[j]-points[j-1])**2).sum())
  total=s[-1];cursor=1
  for j in range(1,8):
   q=j/8.;target=total*q
   while cursor<len(s)-1 and s[cursor]<target:cursor+=1
   a=(target-s[cursor-1])/max(s[cursor]-s[cursor-1],1e-9)
   for axis in range(2):out[i,(j-1)*2+axis]=(points[cursor-1,axis]*(1-a)+points[cursor,axis]*a-origin[i,axis]*(1-q)-end[i,axis]*q)/1000
  ff=connector/(20/3.6);gps=ff;conf=0.;meters=0.
  for e in edges:ff+=free[e];gps+=times[profiles[i],e];conf+=length[e]*support[profiles[i]//168,e];meters+=length[e]
  out[i,14]=total/1000;out[i,15]=ff/600;out[i,16]=gps/600;out[i,17]=conf/max(meters,1);out[i,18]=np.log1p(len(edges));out[i,19]=connector/1000;out[i,20]=1.
 return out

def main():
 start=time.time();run=OUT/'rutas_globales_completas';run.mkdir(exist_ok=False);set_num_threads(4)
 assert json.loads((OUT/'rutas_globales/resultado.json').read_text())['passed_expansion_gate']
 raw=dict(np.load(BASE/'datos/mapa_dirigido_2014.npz'));order=np.argsort(raw['u'],kind='stable').astype(np.int32);ptr=np.r_[0,np.cumsum(np.bincount(raw['u'],minlength=len(raw['xy'])))].astype(np.int64);dest=raw['v'][order];tree=cKDTree(raw['xy']);times=np.load(BASE/'datos/perfiles_mapmatched_2014/time_s.npy',mmap_mode='r');support=np.load(BASE/'datos/perfiles_mapmatched_2014/link_support.npy',mmap_mode='r');names=[f'global_shape_{i}' for i in range(14)]+['global_km','global_freeflow_600s','global_history_600s','global_link_support','global_log_edges','global_connector_km','global_available'];records={}
 for split in ['train','validation']:
  d=dict(np.load(BASE/'datos'/f'{split}_x.npz'));origin=np.column_stack([(d['o_lon']+8.62)*111195*np.cos(np.deg2rad(41.16)),(d['o_lat']-41.16)*111195]);end=np.column_stack([(d['d_lon']+8.62)*111195*np.cos(np.deg2rad(41.16)),(d['d_lat']-41.16)*111195]);_,src=tree.query(origin);_,tar=tree.query(end);dt=pd.to_datetime(d['timestamp'],unit='s',utc=True).tz_convert('Europe/Lisbon');hour=np.asarray(dt.dayofweek)*24+np.asarray(dt.hour);fold=(d['timestamp']//(7*86400))%5 if split=='train' else np.full(len(src),5);profiles=(fold*168+hour).astype(np.int32);X=np.empty((len(src),len(names)),np.float32)
  for first in range(0,len(src),20000):
   last=min(first+20000,len(src));X[first:last]=route_batch(origin[first:last],end[first:last],src[first:last],tar[first:last],profiles[first:last],raw['xy'],ptr,dest,order,raw['length'],raw['seconds'],times,support)
   if first%100000==0 or last==len(src):print(json.dumps(dict(split=split,done=last,total=len(src),seconds=time.time()-start)),flush=True)
   assert time.time()-start<3600,'One-hour route generation limit'
  assert np.isfinite(X).all();path=run/f'{split}_rutas_globales.npz';np.savez_compressed(path,row=d['row'],X=X,names=np.array(names));records[split]=dict(n=len(src),available_fraction=float(X[:,-1].mean()),sha256=sha(path),bytes=path.stat().st_size)
 write(run/'resultado.json',dict(kind='OD-only global expected-time/support route features',records=records,features=names,code_sha256=sha(ROOT.__class__(__file__)),routing_function_sha256=sha(OUT/'rutas_globales.py'),seconds=time.time()-start,Test_access=False,labels_used_for_query_routing=False,profiles='Train: excluded departure-week fold. Validation: full Train only.',new_dense_matrices=0))
if __name__=='__main__':main()
