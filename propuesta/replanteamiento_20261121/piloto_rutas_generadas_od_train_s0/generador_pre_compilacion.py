"""Target-free absorbing directed-road generator with verified numeric tree policy."""
import math,numpy as np
from numba import njit,prange
from ruteo_proyectado import collect_cost_path
@njit(cache=True)
def candidate_features(cur,destination,origin,candidates,static,sx,dx,heading,length,D,chain_dst,chain_src,depart):
 n=len(candidates);f=np.empty((n,49),np.float32);cost=np.empty(n,np.float64)
 for j,candidate in enumerate(candidates):
  if candidate==len(static):cost[j]=0.
  elif candidate==destination:cost[j]=length[candidate]
  else:
   value=length[candidate]+D[chain_dst[candidate],chain_src[destination]]
   cost[j]=value if np.isfinite(value) else 1e7
 smallest=cost.min();gx=sx[destination,0]-dx[cur,0];gy=sx[destination,1]-dx[cur,1];gn=max(math.sqrt(gx*gx+gy*gy),1.);gx/=gn;gy/=gn
 ox=sx[destination,0]-sx[origin,0];oy=sx[destination,1]-sx[origin,1]
 vx=dx[cur,0]-sx[origin,0];vy=dx[cur,1]-sx[origin,1];source_norm=math.sqrt(vx*vx+vy*vy)
 for j,candidate in enumerate(candidates):
  safe=cur if candidate==len(static) else candidate
  for k in range(12):f[j,k]=static[safe,k];f[j,20+k]=static[cur,k];f[j,32+k]=static[destination,k]
  f[j,12]=float(candidate==len(static));f[j,13]=float(candidate==destination);f[j,14]=math.log1p(cost[j]);f[j,15]=math.log1p(max(cost[j]-smallest,0.))
  f[j,16]=heading[safe,0]*heading[cur,1]-heading[safe,1]*heading[cur,0];f[j,17]=heading[safe,0]*heading[cur,0]+heading[safe,1]*heading[cur,1]
  f[j,18]=heading[safe,0]*gx+heading[safe,1]*gy
  qx=dx[safe,0]-sx[destination,0];qy=dx[safe,1]-sx[destination,1];f[j,19]=math.sqrt(qx*qx+qy*qy)/10000
  f[j,44]=ox/10000;f[j,45]=oy/10000;f[j,46]=source_norm/10000;f[j,47]=math.sin(depart);f[j,48]=math.cos(depart)
 return f,cost
@njit(cache=True)
def tree_scores(features,split_feature,border,leaf,scale,bias):
 result=np.empty(len(features),np.float64)
 for i in range(len(features)):
  value=0.
  for t in range(len(leaf)):
   index=0
   for k in range(split_feature.shape[1]):
    if split_feature[t,k]<0:continue
    if features[i,split_feature[t,k]]>border[t,k]:index|=1<<k
   value+=leaf[t,index]
  result[i]=value*scale+bias
 return result
@njit(cache=True)
def edge_at_chain(chain,pos,chain_offsets,prefix,flat_incoming,arc_length):
 a=chain_offsets[chain];b=chain_offsets[chain+1];j=np.searchsorted(prefix[a:b],pos,side='right')+a
 j=max(a+1,min(j,b-1));edge=flat_incoming[j];fraction=(pos-prefix[j-1])/max(arc_length[edge],1e-9)
 return edge,min(max(fraction,0.),1.)
@njit(cache=True)
def append_route(codes,distances,used,chain,dist):
 if dist<=1e-9:return used
 if used>0 and codes[used-1]==chain:distances[used-1]+=dist;return used
 if used>=len(codes):return -1
 codes[used]=chain;distances[used]=dist;return used+1
@njit(cache=True)
def route_descriptor(codes,distances,n,chord,length,free,orientation,speedbin):
 result=np.zeros(90,np.float64);total=distances[:n].sum()
 if n==0 or total<=0:return result
 result[0]=total/1000;result[2]=n/100;result[3]=total/max(chord,1.);result[4]=distances[0]/1000;result[5]=distances[n-1]/1000
 for j in range(n):
  code=codes[j];d=distances[j];w=d/total;speed=length[code]/max(free[code],.001)
  result[1]+=d*free[code]/max(length[code],.001)/600
  result[6]+=w*length[code]/1000;result[7]=max(result[7],d);result[8]+=w*speed/10;result[9]+=w*(orientation[code]%2)
  if code==codes[0]:result[10]+=w
  if code==codes[n-1]:result[11]+=w
  result[12+orientation[code]]+=w;result[20+speedbin[code]]+=w
  result[26+int((np.int64(code)*2654435761)%64)]+=w
 return result
@njit(cache=True)
def generate_one(origin,op,destination,dp,chord,depart,static,sx,dx,heading,length,free,orientation,speedbin,neighbors,neighbor_valid,
 D,conn,routing,u,v,arc_length,chain_src,chain_dst,chain_offsets,prefix,flat_incoming,edge_chain,split_feature,border,leaf,scale,bias,max_steps):
 codes=np.empty(8192,np.int32);distances=np.empty(8192,np.float64);used=0;cur=origin;pos=op;done=False;fallback=False;failed=False
 visited=np.empty(max_steps+1,np.int32);visits=0
 for step in range(max_steps):
  visited[visits]=cur;visits+=1
  arrived=cur==destination and pos<=dp+1e-5
  dist=max((dp if arrived else length[cur])-pos,0.);used=append_route(codes,distances,used,cur,dist)
  if used<0:failed=True;break
  pos=dp if arrived else length[cur]
  if arrived:done=True;break
  candidates=neighbors[cur][neighbor_valid[cur]]
  if not len(candidates):break
  features,cost=candidate_features(cur,destination,origin,candidates,static,sx,dx,heading,length,D,chain_dst,chain_src,depart)
  score=tree_scores(features,split_feature,border,leaf,scale,bias);best=-1;bestscore=-1e300
  for j,candidate in enumerate(candidates):
   nvisit=0
   for k in range(visits):
    if visited[k]==candidate:nvisit+=1
   if nvisit>=2 or cost[j]>=1e7:continue
   if score[j]>bestscore:bestscore=score[j];best=j
  if best<0:break
  cur=candidates[best];pos=0.
 if not done and not failed:
  fallback=True;ea,fa=edge_at_chain(cur,pos,chain_offsets,prefix,flat_incoming,arc_length);ed,fd=edge_at_chain(destination,dp,chain_offsets,prefix,flat_incoming,arc_length)
  ids=np.empty(4096,np.int32);parts=np.empty(4096,np.float64)
  count,cost=collect_cost_path(ea,fa,ed,fd,u,v,arc_length,arc_length,D,conn,routing,ids,parts)
  if count<0 or not np.isfinite(cost):failed=True
  else:
   for j in range(count):
    used=append_route(codes,distances,used,edge_chain[ids[j]],parts[j])
    if used<0:failed=True;break
 if failed or used<=0:return np.zeros(90,np.float32),True,fallback
 return route_descriptor(codes,distances,used,chord,length,free,orientation,speedbin).astype(np.float32),False,fallback
@njit(cache=True,parallel=True)
def generate_queries(chains,fractions,reachability,chords,departures,static,sx,dx,heading,length,free,orientation,speedbin,neighbors,neighbor_valid,
 D,conn,routing,u,v,arc_length,chain_src,chain_dst,chain_offsets,prefix,flat_incoming,edge_chain,split_feature,border,leaf,scale,bias,max_steps):
 n=len(chains);out=np.zeros((n,16,90),np.float32);failed=np.ones((n,16),np.bool_);fallback=np.zeros((n,16),np.bool_)
 for i in prange(n):
  for o in range(4):
   for d in range(4):
    k=o*4+d
    if not reachability[i,o,d]:continue
    origin=int(chains[i,0,o]);destination=int(chains[i,1,d]);op=float(fractions[i,0,o])*length[origin];dp=float(fractions[i,1,d])*length[destination]
    values,bad,tail=generate_one(origin,op,destination,dp,chords[i],departures[i],static,sx,dx,heading,length,free,orientation,speedbin,neighbors,neighbor_valid,
     D,conn,routing,u,v,arc_length,chain_src,chain_dst,chain_offsets,prefix,flat_incoming,edge_chain,split_feature,border,leaf,scale,bias,max_steps)
    out[i,k]=values;failed[i,k]=bad;fallback[i,k]=tail
 return out,failed,fallback
