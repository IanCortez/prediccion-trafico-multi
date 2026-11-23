"""Ordered historical source retrieval with exact safe bounding-box pruning."""
from base_replanteo import *
from numba import njit,prange,set_num_threads
from scipy.spatial import cKDTree
from recuperar_subtrayectos import SubtrajectoryMemory,metric_xy,clock_vars

@njit(parallel=True,cache=True)
def parent_boxes(xy,offset,indexable):
 boxes=np.empty((len(offset)-1,4),np.float32)
 for p in prange(len(boxes)):
  x0=np.inf;y0=np.inf;x1=-np.inf;y1=-np.inf
  for i in range(offset[p],offset[p+1]):
   if not indexable[i]:continue
   x0=min(x0,xy[i,0]);y0=min(y0,xy[i,1]);x1=max(x1,xy[i,0]);y1=max(y1,xy[i,1])
  boxes[p,0]=x0;boxes[p,1]=y0;boxes[p,2]=x1;boxes[p,3]=y1
 return boxes

@njit(cache=True)
def best_ordered_interval(p,offset,xy,indexable,q):
 a=offset[p];n=offset[p+1]-a
 od=np.full(n,np.inf,np.float64);dd=np.full(n,np.inf,np.float64)
 for i in range(n):
  if not indexable[a+i]:continue
  dx=xy[a+i,0]-q[0];dy=xy[a+i,1]-q[1];od[i]=dx*dx+dy*dy
  dx=xy[a+i,0]-q[2];dy=xy[a+i,1]-q[3];dd[i]=dx*dx+dy*dy
 deque=np.empty(n,np.int64);head=0;tail=0;best=np.inf;bi=-1;bj=-1;blen=0.
 for j in range(4,n):
  new=j-4
  while tail>head and od[deque[tail-1]]>od[new]:tail-=1
  deque[tail]=new;tail+=1
  while tail>head and deque[head]<j-119:head+=1
  if not indexable[a+j]:continue
  i=deque[head]
  if od[i]+dd[j]>best+1e-10:continue
  dx=xy[a+j,0]-xy[a+i,0];dy=xy[a+j,1]-xy[a+i,1]
  length=np.sqrt(dx*dx+dy*dy);span=j-i
  if indexable[a+i] and length>=200 and length/(15*span)*3.6<=120:
   value=od[i]+dd[j]
   if value<best-1e-10 or (abs(value-best)<=1e-10 and (i<bi or(i==bi and j>bj))):
    best=value;bi=i;bj=j;blen=length
  else:
   for i in range(max(0,j-119),j-3):
    if not indexable[a+i]:continue
    value=od[i]+dd[j]
    if value>best+1e-10:continue
    dx=xy[a+j,0]-xy[a+i,0];dy=xy[a+j,1]-xy[a+i,1]
    length=np.sqrt(dx*dx+dy*dy);span=j-i
    if length<200 or length/(15*span)*3.6>120:continue
    if value<best-1e-10 or (abs(value-best)<=1e-10 and (i<bi or(i==bi and j>bj))):
     best=value;bi=i;bj=j;blen=length
 return best,bi,bj,blen

@njit(parallel=True,cache=True)
def ordered_candidates(oi,di,parent_of_point,offset,n,ts,fold,xy,indexable,boxes,qxy,qts,qfold,starts,ends,prune=True):
 B,K=oi.shape
 parent=np.full((B,64),-1,np.int32);first=np.zeros((B,64),np.uint8);last=np.zeros((B,64),np.uint8)
 duration=np.zeros((B,64),np.float32);weight=np.full((B,64),-1e30,np.float64)
 error=np.full((B,64),np.inf,np.float32);full=np.zeros((B,64),np.bool_)
 evaluated=np.zeros(B,np.int32);discovered=np.zeros(B,np.int32);minerror=np.full(B,np.inf,np.float64)
 for q in prange(B):
  raw=np.sort(np.concatenate((parent_of_point[oi[q]],parent_of_point[di[q]])))
  candidates=np.empty(len(raw),np.int32);m=0;previous=-1
  for p in raw:
   if p!=previous:
    previous=p
    if qfold[q]<5 and fold[p]==qfold[q]:continue
    candidates[m]=p;m+=1
  discovered[q]=m;lower=np.empty(m,np.float64)
  for j in range(m):
   p=candidates[j];value=0.
   for k in range(2):
    dx=max(0.,boxes[p,0]-qxy[q,2*k],qxy[q,2*k]-boxes[p,2])
    dy=max(0.,boxes[p,1]-qxy[q,2*k+1],qxy[q,2*k+1]-boxes[p,3])
    value+=dx*dx+dy*dy
   lower[j]=value
  order=np.argsort(lower);qh,qw=clock_vars(qts[q],starts,ends)
  qlength=np.sqrt((qxy[q,2]-qxy[q,0])**2+(qxy[q,3]-qxy[q,1])**2)
  kept=0
  for ix in order:
   worst=0
   for k in range(1,64):
    if weight[q,k]<weight[q,worst] or(weight[q,k]==weight[q,worst] and parent[q,k]>parent[q,worst]):worst=k
   # Calendar and weekend logpenalties <=0, so this spatial upper bound is safe.
   if prune and kept==64 and -.5*lower[ix]/2500<weight[q,worst]-1e-12:break
   p=candidates[ix];evaluated[q]+=1
   best,i,j,length=best_ordered_interval(p,offset,xy,indexable,qxy[q])
   if i<0:continue
   minerror[q]=min(minerror[q],np.sqrt(best))
   sh,sw=clock_vars(ts[p]+15*i,starts,ends);delta=(sh-qh+12)%24-12
   logweight=-.5*best/2500-.5*(delta/4)**2+(0. if sw==qw else np.log(.35))
   if logweight>weight[q,worst] or(logweight==weight[q,worst] and p<parent[q,worst]):
    if parent[q,worst]<0:kept+=1
    parent[q,worst]=p;first[q,worst]=i;last[q,worst]=j
    duration[q,worst]=15*(j-i)*min(2.,max(.5,qlength/max(length,1.)))
    weight[q,worst]=logweight;error[q,worst]=np.sqrt(best);full[q,worst]=(i==0 and j==n[p]-1)
  # Stable canonical ordering: descending logweight, then ascending parentID.
  index=np.arange(64)
  for a in range(1,64):
   b=a
   while b>0:
    x=index[b];y=index[b-1]
    if weight[q,x]>weight[q,y] or(weight[q,x]==weight[q,y] and parent[q,x]<parent[q,y]):
     index[b]=y;index[b-1]=x;b-=1
    else:break
  pp=parent[q].copy();ff=first[q].copy();ll=last[q].copy();tt=duration[q].copy();ww=weight[q].copy();ee=error[q].copy();uu=full[q].copy()
  for a in range(64):
   k=index[a];parent[q,a]=pp[k];first[q,a]=ff[k];last[q,a]=ll[k];duration[q,a]=tt[k];weight[q,a]=ww[k];error[q,a]=ee[k];full[q,a]=uu[k]
 return parent,first,last,duration,weight,error,full,evaluated,discovered,minerror

class OrderedSourceMemory(SubtrajectoryMemory):
 def __init__(self,source_arrival_before=None):
  super().__init__();z=self.data
  assert z['parent_gps_n'].max()<=120
  self.available=np.zeros(len(z['parent_gps_n']),bool);self.available[z['eligible']]=True
  if source_arrival_before is not None:self.available&=z['timestamp']+z['time_s']<int(source_arrival_before)
  self.boxes=parent_boxes(z['xy_m'],z['offsets'],self.indexable)
 def build_tree(self,fold=5):
  assert 0<=fold<=5;z=self.data
  selected=self.indexable&self.available[self.parent]
  if fold<5:selected&=z['parent_fold'][self.parent]!=fold
  ids=np.flatnonzero(selected)
  assert self.available[self.parent[ids]].all()
  if fold<5:assert np.all(z['parent_fold'][self.parent[ids]]!=fold)
  self.tree=None;self.tree_ids=None
  self.tree_ids=ids;self.tree=cKDTree(z['xy_m'][ids].astype(np.float64));self.tree_fold=fold
 def query(self,d,training=False,k=2048,diagnostics=False,prune=True):
  from consulta_od import od_records
  assert set(d)=={'o_lon','o_lat','d_lon','d_lat','timestamp'} and k==2048
  q=od_records([{key:int(d[key][i]) if key=='timestamp' else float(d[key][i]) for key in d} for i in range(len(d['timestamp']))])
  qxy=metric_xy(q);qfold=(q['timestamp']//604800)%5 if training else np.full(len(qxy),5)
  assert len(np.unique(qfold))==1 and self.tree_fold==int(qfold[0])
  z=self.data
  if not training:assert np.max(z['timestamp'][self.available]+z['time_s'][self.available])<q['timestamp'].min()
  _,oi=self.tree.query(qxy[:,:2],k=k,workers=2);_,di=self.tree.query(qxy[:,2:],k=k,workers=2)
  result=ordered_candidates(self.tree_ids[oi],self.tree_ids[di],self.parent,z['offsets'],z['parent_gps_n'],z['timestamp'],z['parent_fold'],z['xy_m'],self.indexable,self.boxes,qxy,q['timestamp'],qfold,self.starts,self.ends,prune)
  return result if diagnostics else result[:7]
