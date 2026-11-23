"""Exact bounded-window ordered endpoint match inside discovered historical parents."""
import numpy as np
from numba import njit,prange

@njit(parallel=True,cache=True)
def ordered_parent_support(oi,di,parent_of_point,offset,xy,qxy):
 B,K=oi.shape;counts=np.zeros(B,np.int32);minimum=np.full(B,np.inf,np.float64)
 for q in prange(B):
  discovered=np.sort(np.concatenate((parent_of_point[oi[q]],parent_of_point[di[q]])))
  previous=-1
  for ix in range(len(discovered)):
   parent=discovered[ix]
   if parent==previous:continue
   previous=parent;a=offset[parent];n=offset[parent+1]-a
   od=np.empty(n,np.float64);dd=np.empty(n,np.float64)
   for i in range(n):
    dx=xy[a+i,0]-qxy[q,0];dy=xy[a+i,1]-qxy[q,1];od[i]=dx*dx+dy*dy
    dx=xy[a+i,0]-qxy[q,2];dy=xy[a+i,1]-qxy[q,3];dd[i]=dx*dx+dy*dy
   deque=np.empty(n,np.int64);head=0;tail=0;best=np.inf
   for last in range(4,n):
    new=last-4
    while tail>head and od[deque[tail-1]]>od[new]:tail-=1
    deque[tail]=new;tail+=1
    while tail>head and deque[head]<last-119:head+=1
    first=deque[head]
    # If even the minimum origin-distance cannot improve this parent, skip this end.
    if od[first]+dd[last]>best:continue
    dx=xy[a+last,0]-xy[a+first,0];dy=xy[a+last,1]-xy[a+first,1]
    length=np.sqrt(dx*dx+dy*dy);span=last-first
    if length>=200 and length/(15*span)*3.6<=120:
     value=od[first]+dd[last]
     if value<best:best=value
    else:
     # Exact fallback for physical constraints; never return an invalid unconstrained minimum.
     for first in range(max(0,last-119),last-3):
      value=od[first]+dd[last]
      if value>best:continue
      dx=xy[a+last,0]-xy[a+first,0];dy=xy[a+last,1]-xy[a+first,1]
      length=np.sqrt(dx*dx+dy*dy);span=last-first
      if length>=200 and length/(15*span)*3.6<=120 and value<best:best=value
   if np.isfinite(best):
    counts[q]+=1
    if best<minimum[q]:minimum[q]=best
 return counts,np.sqrt(minimum)

def brute_parent_support(oi,di,parent_of_point,offset,xy,qxy):
 counts=[];errors=[]
 for q in range(len(oi)):
  found=0;minimum=np.inf
  for parent in np.unique(np.r_[parent_of_point[oi[q]],parent_of_point[di[q]]]):
   p=xy[offset[parent]:offset[parent+1]].astype(np.float64)
   od=((p-qxy[q,:2])**2).sum(1);dd=((p-qxy[q,2:])**2).sum(1);best=np.inf
   for j in range(4,len(p)):
    first=np.arange(max(0,j-119),j-3);span=j-first
    length=np.sqrt(((p[j]-p[first])**2).sum(1))
    valid=(length>=200)&(length/(15*span)*3.6<=120)
    if valid.any():best=min(best,float((od[first[valid]]+dd[j]).min()))
   if np.isfinite(best):found+=1;minimum=min(minimum,best)
  counts.append(found);errors.append(np.sqrt(minimum))
 return np.asarray(counts),np.asarray(errors)
