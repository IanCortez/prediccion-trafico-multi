"""Fold-safe empiricalroadstate kernel with fixed15s finite-duration endpoint inference."""
import numpy as np,pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from collections import OrderedDict
from conteos_red_cinetica import select_fold,decode_pairs,position_counts
from coordenadas_trayecto_2d import metric_queries
class RoadStateGeometry:
 def __init__(self,path):
  self.raw=dict(np.load(path));r=self.raw
  self.bins=np.maximum(1,np.ceil(r['length']/50).astype(np.int64))
  self.offsets=np.r_[0,np.cumsum(self.bins)]
 def centres(self,nodes,kind):
  spatial=np.asarray(nodes)//(3 if kind=='kinetic' else 1)
  edge=np.searchsorted(self.offsets,spatial,side='right')-1
  assert np.all((edge>=0)&(edge<len(self.bins)))
  fraction=(spatial-self.offsets[edge]+.5)/self.bins[edge]
  r=self.raw;a=r['xy'][r['u'][edge]];b=r['xy'][r['v'][edge]]
  return a+(b-a)*fraction[:,None],np.maximum(r['length'][edge]/self.bins[edge],1e-6)
def load_counts(path):
 with np.load(path) as z:
  return {key:z[key].copy() for key in ['kinetic_pair_keys','kinetic_pair_counts','stop_keys','stop_counts','start_keys','start_counts','kinetic_state_count']}
def source_events(keys,weights,state_count,excluded_fold,kind):
 keys,weights=select_fold(keys,weights,state_count,excluded_fold,pairs=False)
 context=(keys//np.uint64(state_count)).astype(np.int16)
 nodes=(keys%np.uint64(state_count)).astype(np.int64)
 if kind=='position':nodes=nodes//3
 return context%168,nodes,weights.astype(np.float64)
class RoadClockKernel:
 def __init__(self,counts,geometry,kind='kinetic',excluded_fold=5,alpha=10.,cache_limit=4):
  assert kind in ['kinetic','position'] and excluded_fold in range(6)
  self.kind=kind;self.excluded_fold=excluded_fold;self.alpha=float(alpha);self.cache_limit=cache_limit
  K=int(counts['kinetic_state_count'])
  keys,weights=select_fold(counts['kinetic_pair_keys'],counts['kinetic_pair_counts'],K,excluded_fold)
  if kind=='position':
   keys,weights=position_counts(keys,weights,K);state_count=K//3
  else:state_count=K
  context,source,destination=decode_pairs(keys,state_count)
  self.hour=context%168;self.weight=weights.astype(np.float64)
  stop_hour,stop_node,stop_weight=source_events(counts['stop_keys'],counts['stop_counts'],K,excluded_fold,kind)
  start_hour,start_node,start_weight=source_events(counts['start_keys'],counts['start_counts'],K,excluded_fold,kind)
  # This catalogue and everymeasurement-support lookup are constructed AFTER sourcefold removal.
  self.nodes=np.unique(np.r_[source,destination,stop_node,start_node]);self.size=len(self.nodes)
  self.source=np.searchsorted(self.nodes,source);self.destination=np.searchsorted(self.nodes,destination)
  self.stop_node=np.searchsorted(self.nodes,stop_node);self.stop_hour=stop_hour;self.stop_weight=stop_weight
  self.start_node=np.searchsorted(self.nodes,start_node);self.start_hour=start_hour;self.start_weight=start_weight
  self.centres,self.width=geometry.centres(self.nodes,kind)
  self.tree=cKDTree(self.centres)
  C=sparse.csr_matrix((self.weight,(self.source,self.destination)),shape=(self.size,self.size),dtype=np.float64)
  C.sum_duplicates();C.sort_indices()
  self.stop_global=np.bincount(self.stop_node,weights=stop_weight,minlength=self.size)
  self.start_global=np.bincount(self.start_node,weights=start_weight,minlength=self.size)
  total=np.asarray(C.sum(1)).ravel()+self.stop_global
  inv=np.divide(1.,total,out=np.zeros_like(total),where=total>0)
  self.global_P=C.multiply(inv[:,None]).tocsr();self.global_P.sort_indices()
  self.global_r=self.stop_global*inv
  spatial=self.nodes//(3 if kind=='kinetic' else 1)
  _,back,multiplicity=np.unique(spatial,return_inverse=True,return_counts=True)
  geometric=self.width/multiplicity[back] if self.size else np.empty(0)
  geometric=geometric/geometric.sum() if self.size else geometric
  self.global_start_prior=(self.start_global+geometric)/(self.start_global.sum()+1)
  self.cache=OrderedDict();self.cache_peak=0;self.row_mass_max=0.
  self.global_dead_rows=int(np.sum(total==0))
 def kernel(self,hour):
  hour=int(hour);assert 0<=hour<168
  if hour in self.cache:
   self.cache.move_to_end(hour);return self.cache[hour]
  selected=self.hour==hour
  C=sparse.csr_matrix((self.weight[selected],(self.source[selected],self.destination[selected])),shape=(self.size,self.size))
  C.sum_duplicates();C.sort_indices()
  stopped=np.bincount(self.stop_node[self.stop_hour==hour],weights=self.stop_weight[self.stop_hour==hour],minlength=self.size)
  starts=np.bincount(self.start_node[self.start_hour==hour],weights=self.start_weight[self.start_hour==hour],minlength=self.size)
  total=np.asarray(C.sum(1)).ravel()+stopped
  inverse=1/(total+self.alpha);mix=self.alpha*inverse
  Q=(C.multiply(inverse[:,None])+self.global_P.multiply(mix[:,None])).tocsr()
  Q.sum_duplicates();Q.sort_indices()
  stop=(stopped+self.alpha*self.global_r)*inverse
  initial=(starts+self.alpha*self.global_start_prior)/(starts.sum()+self.alpha)
  mass=np.asarray(Q.sum(1)).ravel()+stop
  assert np.isfinite(Q.data).all() and np.isfinite(stop).all() and (Q.data>=0).all() and (stop>=0).all()
  assert mass.max(initial=0)<=1+1e-12
  self.row_mass_max=max(self.row_mass_max,float(mass.max(initial=0)))
  value=(Q.transpose().tocsr(),stop,initial)
  self.cache[hour]=value;self.cache.move_to_end(hour)
  if len(self.cache)>self.cache_limit:self.cache.popitem(last=False)
  self.cache_peak=max(self.cache_peak,len(self.cache))
  return value
 def measurement(self,point):
  index=np.asarray(sorted(self.tree.query_ball_point(point,90.)),dtype=np.int64)
  delta=self.centres[index]-point
  weight=np.exp(-.5*np.square(delta).sum(1)/(30.*30.))
  return index,weight
 def one(self,origin,destination,hours):
  assert len(hours)==120
  oi,ow=self.measurement(origin);di,dw=self.measurement(destination)
  result=np.zeros(116,np.float64)
  diagnostics=dict(origin_nodes=len(oi),destination_nodes=len(di),initial_normalizer=0.,support=0.)
  if not len(oi) or not len(di):return result,diagnostics
  _,_,prior=self.kernel(hours[0]);initial_weight=prior[oi]*ow
  normalizer=float(initial_weight.sum());diagnostics['initial_normalizer']=normalizer
  if normalizer<=0:return result,diagnostics
  flow=np.zeros(self.size,np.float64);flow[oi]=initial_weight/normalizer
  for n in range(120):
   QT,stop,_=self.kernel(hours[n])
   if n>=4:result[n-4]=float(flow[di]@(stop[di]*dw))
   if n<119:flow=QT@flow
  total=float(result.sum());diagnostics['support']=total
  assert np.isfinite(result).all() and (result>=0).all() and total<=1+1e-10
  return result,diagnostics
 def predict(self,queries):
  xy=metric_queries(queries);B=len(xy)
  timestamp=np.asarray(queries['timestamp'],np.int64)
  clock=timestamp[:,None]+15*np.arange(120)[None]
  local=pd.to_datetime(clock.ravel(),unit='s',utc=True).tz_convert('Europe/Lisbon')
  hours=(np.asarray(local.dayofweek)*24+np.asarray(local.hour)).reshape(B,120)
  weights=np.zeros((B,116),np.float64);diagnostics=[]
  for i in range(B):
   weights[i],detail=self.one(xy[i,:2],xy[i,2:],hours[i]);diagnostics.append(detail)
  support=weights.sum(1);positive=support>0
  probability=np.divide(weights,support[:,None],out=np.zeros_like(weights),where=support[:,None]>0)
  return dict(weights=weights,probability=probability,supported=positive,support=support,diagnostics=diagnostics)
