"""Ordered public-route edge tokens, independent of observed trajectory or duration."""
import numpy as np
from numba import njit,prange
from consulta_historicos_cruzados import prepare
from map_matching import edge_distance
from ruteo_proyectado import collect_cost_path
@njit(cache=True,parallel=True)
def sample_public_routes(origin,destination,od_m,oe,of,od,de,df,dd,u,v,length,free,arc_cost,D,conn,routing,is_free):
 n=len(origin);tokens=np.zeros((n,32),np.int32);summary=np.zeros((n,4),np.float32);errors=np.zeros(n,np.float32)
 for block in prange(64):
  ids=np.empty(4096,np.int32);parts=np.empty(4096,np.float64)
  for q in range(block*n//64,(block+1)*n//64):
   best=np.inf;bo=-1;bd=-1
   for a in range(4):
    for b in range(4):
     if od[q,a]>500 or dd[q,b]>500:continue
     cost=edge_distance(oe[q,a],of[q,a],de[q,b],df[q,b],u,v,arc_cost,D,*conn)
     if not np.isfinite(cost):continue
     snap=od[q,a]+dd[q,b];score=cost+snap/(10/3.6) if is_free else cost+3*snap
     if score<best:best=score;bo=a;bd=b
   if bo<0:continue
   used,cost=collect_cost_path(oe[q,bo],of[q,bo],de[q,bd],df[q,bd],u,v,length,arc_cost,D,conn,routing,ids,parts)
   if used<=0:continue
   total=0.;ff=0.;planning=0.
   for j in range(used):
    total+=parts[j];ff+=free[ids[j]]*parts[j]/length[ids[j]];planning+=arc_cost[ids[j]]*parts[j]/length[ids[j]]
   errors[q]=abs(planning-cost)
   if total<=1e-8:continue
   snap=od[q,bo]+dd[q,bd];summary[q,0]=(total+snap)/1000;summary[q,1]=(ff+snap/(20/3.6))/600;summary[q,2]=1.;summary[q,3]=used
   cursor=0;cumulative=parts[0]
   for j in range(32):
    target=total*(j+.5)/32
    while cursor<used-1 and cumulative<target:
     cursor+=1;cumulative+=parts[cursor]
    tokens[q,j]=ids[cursor]+1
 return tokens,summary,errors
def build_public_sequences(items):
 p=prepare(items);r=p['raw'];oe,of,od,de,df,dd=p['projected'];all_tokens=[];all_summary=[]
 for kind in ['length','free']:
  D,conn,routing=p['routes'][kind]
  ids,summary,error=sample_public_routes(p['origin'],p['destination'],p['d']['od_m'],oe,of,od,de,df,dd,r['u'],r['v'],r['length'],r['seconds'],r['seconds'] if kind=='free' else r['length'],D,conn,routing,kind=='free')
  assert float(error.max())<.1 and ids.min()>=0 and ids.max()<=len(r['u'])
  all_tokens.append(ids);all_summary.append(summary)
 return np.stack(all_tokens,1),np.stack(all_summary,1)
