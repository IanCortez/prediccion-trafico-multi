"""Eight public route hypotheses; explicit uncertainty in endpoint map attachment."""
import numpy as np
from numba import njit,prange
from consulta_historicos_cruzados import prepare
from map_matching import edge_distance
from ruteo_proyectado import collect_cost_path,path_shape
@njit(cache=True,parallel=True)
def choose_pairs(oe,of,od,de,df,dd,u,v,length,free,DL,CL,DF,CF):
 n=len(oe);pairs=np.full((n,8,2),-1,np.int8)
 for q in prange(n):
  costs=np.full((2,16),np.inf);snap2=np.full(16,np.inf);best=np.full(2,-1,np.int32);bestscore=np.full(2,np.inf)
  for a in range(4):
   for b in range(4):
    k=a*4+b
    if od[q,a]>500 or dd[q,b]>500:continue
    for kind in range(2):
     if kind==0:cost=edge_distance(oe[q,a],of[q,a],de[q,b],df[q,b],u,v,length,DL,*CL)
     else:cost=edge_distance(oe[q,a],of[q,a],de[q,b],df[q,b],u,v,free,DF,*CF)
     costs[kind,k]=cost
     score=cost+(3*(od[q,a]+dd[q,b]) if kind==0 else (od[q,a]+dd[q,b])/(10/3.6))
     if score<bestscore[kind]:bestscore[kind]=score;best[kind]=k
    if np.isfinite(costs[0,k]) or np.isfinite(costs[1,k]):snap2[k]=od[q,a]**2+dd[q,b]**2
  for kind in range(2):
   k=best[kind]
   if k>=0:pairs[q,kind,0]=k//4;pairs[q,kind,1]=k%4
  selected=np.full(3,-1,np.int32)
  for rank in range(3):
   winner=-1;score=np.inf
   for k in range(16):
    if not np.isfinite(snap2[k]):continue
    a=k//4;b=k%4;duplicate=False
    for h in range(2+rank*2):
     pa=pairs[q,h,0];pb=pairs[q,h,1]
     if pa>=0 and oe[q,pa]==oe[q,a] and de[q,pb]==de[q,b] and of[q,pa]==of[q,a] and df[q,pb]==df[q,b]:duplicate=True;break
    if not duplicate and snap2[k]<score:winner=k;score=snap2[k]
   if winner<0:break
   selected[rank]=winner
   for kind in range(2):
    slot=2+2*rank+kind;pairs[q,slot,0]=winner//4;pairs[q,slot,1]=winner%4
 return pairs
@njit(cache=True,parallel=True)
def build_kind(origin,destination,oe,of,od,de,df,dd,pairs,profile,u,v,length,free,xy,arc_cost,D,conn,routing,profile_time,state_support,link_support,kind):
 n=len(origin);out=np.zeros((n,8,31),np.float32);errors=np.zeros(n,np.float32)
 for block in prange(64):
  ids=np.empty(4096,np.int32);parts=np.empty(4096,np.float64);vertices=np.empty((4099,2),np.float64);cumulative=np.empty(4099,np.float64)
  for q in range(block*n//64,(block+1)*n//64):
   bestscore=np.inf
   a0=pairs[q,kind,0];b0=pairs[q,kind,1]
   if a0>=0:
    bestcost=edge_distance(oe[q,a0],of[q,a0],de[q,b0],df[q,b0],u,v,arc_cost,D,*conn)
    bestscore=bestcost+(3*(od[q,a0]+dd[q,b0]) if kind==0 else (od[q,a0]+dd[q,b0])/(10/3.6))
   for slot in range(kind,8,2):
    a=pairs[q,slot,0];b=pairs[q,slot,1]
    if a<0:continue
    ea=oe[q,a];eb=de[q,b];fa=of[q,a];fb=df[q,b]
    used,cost=collect_cost_path(ea,fa,eb,fb,u,v,length,arc_cost,D,conn,routing,ids,parts)
    if used<0 or not np.isfinite(cost):continue
    S=np.zeros(26,np.float32);gps=0.;ff=0.;state=0.;link=0.;total=0.;planning=0.;p=profile[q];fold=p//168
    for j in range(used):
     edge=ids[j];part=parts[j];ratio=part/length[edge];total+=part;gps+=profile_time[p,edge]*ratio;ff+=free[edge]*ratio
     state+=part*state_support[p,edge];link+=part*link_support[fold,edge];planning+=arc_cost[edge]*ratio
    projected=xy[u[ea]]+(xy[v[ea]]-xy[u[ea]])*fa
    distance=path_shape(origin[q],destination[q],projected,ids,parts,used,xy,u,v,length,vertices,cumulative,S)
    end_projected=xy[u[eb]]+(xy[v[eb]]-xy[u[eb]])*fb;snap=od[q,a]+dd[q,b]
    error=max(abs(planning-cost),np.sqrt(((vertices[used+1]-end_projected)**2).sum()),abs(distance-total-snap));errors[q]=max(errors[q],error)
    gps+=snap/(20/3.6);ff+=snap/(20/3.6)
    S[21]=ff/600;S[22]=od[q,a]/1000;S[23]=dd[q,b]/1000
    score=cost+(3*snap if kind==0 else snap/(10/3.6))
    S[24]=max(0.,(score-bestscore)/(1000 if kind==0 else 600)) if np.isfinite(bestscore) else 0.
    S[25]=1.
    out[q,slot,:26]=S;out[q,slot,26]=gps/600;out[q,slot,27]=state/max(total,1);out[q,slot,28]=link/max(total,1)
    out[q,slot,29]=np.log(max(gps/max(ff,1),.01));out[q,slot,30]=np.log(max(1000*gps/max(distance,1),.01))
 return out,errors
def build_uncertain_routes(items,training=False):
 p=prepare(items);r=p['raw'];oe,of,od,de,df,dd=p['projected']
 DL,CL,RL=p['routes']['length'];DF,CF,RF=p['routes']['free']
 pairs=choose_pairs(oe,of,od,de,df,dd,r['u'],r['v'],r['length'],r['seconds'],DL,CL,DF,CF)
 fold=(p['d']['timestamp']//604800)%5 if training else np.full(len(items),5)
 profile=(fold*168+p['clock']).astype(np.int32);out=np.zeros((len(items),8,31),np.float32)
 for kind,(D,conn,routing) in enumerate([(DL,CL,RL),(DF,CF,RF)]):
  bank,error=build_kind(p['origin'],p['destination'],oe,of,od,de,df,dd,pairs,profile,r['u'],r['v'],r['length'],r['seconds'],r['xy'],r['length'] if kind==0 else r['seconds'],D,conn,routing,*p['profiles'],kind)
  assert float(error.max())<.1
  out+=bank
 assert np.isfinite(out).all()
 return out,pairs
