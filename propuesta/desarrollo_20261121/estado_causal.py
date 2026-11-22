"""Prequential traffic-state diagnostic. New online protocol, never static confirmation."""
from piloto_transferencia import ROOT,BASE,OUT,write,metric,paired,sha
import numpy as np,heapq,time,json,bisect
from collections import deque

class Window:
 def __init__(self,seconds):self.seconds=seconds;self.queues={};self.values={}
 def add(self,key,t,r):
  q=self.queues.setdefault(key,deque());a=self.values.setdefault(key,[]);q.append((t,r));bisect.insort(a,r)
 def get(self,key,t):
  q=self.queues.get(key)
  if q is None:return 0.,0
  a=self.values[key]
  while q and q[0][0]<t-self.seconds:
   _,v=q.popleft();i=bisect.bisect_left(a,v);assert i<len(a) and a[i]==v;a.pop(i)
  n=len(a)
  if not n:return 0.,0
  return float((a[(n-1)//2]+a[n//2])/2),n

def keys(d):
 x=(d['o_lon']+8.62)*111195*np.cos(np.deg2rad(41.16));y=(d['o_lat']-41.16)*111195;dx=(d['d_lon']+8.62)*111195*np.cos(np.deg2rad(41.16));dy=(d['d_lat']-41.16)*111195
 dist=np.digitize(d['od_m'],[1000,2000,4000,8000]);direction=(np.arctan2(dy-y,dx-x)*2/np.pi).astype(int)
 return [(('global',),('origin',int(a//1000),int(b//1000)),('destination',int(c//1000),int(e//1000)),('movement',int(s),int(h))) for a,b,c,e,s,h in zip(x,y,dx,dy,dist,direction)]

def simulate(t,y,p,k):
 state={h:Window(h*3600) for h in (2,6,24)};events=[];out=np.empty((len(t),5));count=np.zeros(len(t),np.int32);latest=-1;max_used=-1
 for i,now in enumerate(t):
  while events and events[0][0]<now:
   available,j=heapq.heappop(events);latest=max(latest,available);r=float(y[j]-p[j])
   for s in state.values():
    for key in k[j]:s.add(key,available,r)
  assert latest<now
  max_used=max(max_used,latest);m,n=state[2].get(k[i][0],now);count[i]=n;global_bias=m*n/(n+50)
  out[i,0]=p[i];out[i,1]=p[i]+np.clip(global_bias,-120,120)
  for col,h in enumerate((2,6,24),2):
   s=state[h];glob,ng=s.get(k[i][0],now);local=[];support=[]
   for key in k[i][1:]:
    med,num=s.get(key,now);local.append(med*num/(num+30));support.append(num/(num+30))
   # Shrink local regimes toward city-wide bias when support is sparse.
   denom=sum(support);bias=(sum(local)+(1-min(denom/3,1))*glob*ng/(ng+50))/(denom+1-min(denom/3,1))
   out[i,col]=p[i]+np.clip(bias,-120,120)
  out[i]=np.clip(out[i],60,1785)
  # The duration is used only to schedule when the completed trip becomes observable.
  heapq.heappush(events,(int(now)+float(y[i])+60,i))
 return out,count

def main():
 start=time.time();run=OUT/'estado_causal';run.mkdir(exist_ok=False)
 protocol=dict(kind='New online exploratory protocol, not the frozen static OD benchmark',label_availability='departure + observed duration + 60 seconds; strictly before current departure',initial_state='empty; no Validation future and no Train in-sample residual warm start',fixed_configurations=['frozen_static','global_2h','local_2h','local_6h','local_24h'],fit_or_tuning='none; fixed parameters before simulation',limits=dict(max_bias_s=120,local_shrinkage_n=30,global_shrinkage_n=50),scope='all existing 127941 Validation trips in original order; Test remains sealed')
 write(run/'protocolo.json',protocol)
 d=dict(np.load(BASE/'datos/validation_x.npz'));z=np.load(BASE/'experimentos/od_mezcla8_geometria_conjunta_s0_fp32_v2/validacion.npz');y=z['true'].astype(float);p=z['pred'].astype(float);t=d['timestamp'];assert np.array_equal(d['row'],z['row']) and np.all(np.diff(t)>=0);k=keys(d)
 predictions,count=simulate(t,y,p,k)
 # Future label perturbations cannot affect earlier predictions.
 n=min(4000,len(y));cut=2000;y2=y[:n].copy();y2[cut:]=60
 check,_=simulate(t[:n],y[:n],p[:n],k[:n]);changed,_=simulate(t[:n],y2,p[:n],k[:n]);assert np.array_equal(check[:cut],changed[:cut]);assert check[cut,1:].tolist()==changed[cut,1:].tolist()
 names=protocol['fixed_configurations'];metrics={name:metric(y,predictions[:,i]) for i,name in enumerate(names)};deltas={name:paired(y,predictions[:,i],p,t) for i,name in enumerate(names) if i};np.savez_compressed(run/'predicciones.npz',row=d['row'],timestamp=t,true=y,**{name:predictions[:,i] for i,name in enumerate(names)},history_n=count)
 result=dict(protocol=protocol,metrics=metrics,paired_vs_static=deltas,future_label_invariance_passed=True,mean_2h_history_n=float(count.mean()),seconds=time.time()-start,code_sha256=sha(ROOT.__class__(__file__)),eligible_static_goal=False,Test_access=False,conclusion='Nueva informacion de viajes completados; no comparar como si fuese el mismo contrato estatico. No se selecciona ni certifica un modelo con Test.')
 write(run/'resultado.json',result);print(json.dumps(result,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else str(v)),flush=True)
if __name__=='__main__':main()
