"""Publicraw5 roadattachments; FITGPS mapmatching with preserved15second sourceclock."""
from base_replanteo import *
from scipy.spatial import cKDTree
from numba import njit,prange,set_num_threads
from map_matching import project_candidates,node_connections,viterbi_match,edge_distance
from coordenadas_trayecto_2d import metric_queries
class PublicRoadSampler:
 def __init__(self):
  set_num_threads(2)
  self.raw=dict(np.load(DATA/'mapa_dirigido_2014.npz'))
  self.comp=dict(np.load(DATA/'grafo_compacto_2014.npz'))
  info=read(DATA/'distancias_2014/manifiesto.json')
  assert sha(DATA/'mapa_dirigido_2014.npz')==info['raw_graph_sha256']
  self.D=np.load(DATA/'distancias_2014/D.npy',mmap_mode='r');self.connections=node_connections(self.comp)
  raw=self.raw;count=np.maximum(1,np.ceil(raw['length']/25).astype(np.int32))
  self.sample_edges=np.repeat(np.arange(len(count),dtype=np.int32),count)
  index=np.arange(len(self.sample_edges))-np.repeat(np.cumsum(count)-count,count)
  fraction=(index+.5)/count[self.sample_edges]
  a=raw['xy'][raw['u'][self.sample_edges]];b=raw['xy'][raw['v'][self.sample_edges]]
  self.tree=cKDTree(a+(b-a)*fraction[:,None])
  self.bins=np.maximum(1,np.ceil(raw['length']/50).astype(np.int64))
  self.state_offsets=np.r_[0,np.cumsum(self.bins)]
 def candidates(self,points):
  points=np.ascontiguousarray(points,dtype=np.float64)
  _,near=self.tree.query(points,k=32,workers=2)
  r=self.raw
  return project_candidates(points,near,self.sample_edges,r['xy'],r['u'],r['v'])
 def public_attachments(self,queries):
  xy=metric_queries(queries)
  return self.candidates(xy[:,:2])+self.candidates(xy[:,2:])
 def match_fit(self,points,fit_offsets):
  edge,fraction,error=self.candidates(points);r=self.raw
  matched=viterbi_match(points,fit_offsets,edge,fraction,error,r['u'],r['v'],r['length'],self.D,*self.connections)
  return matched,edge[:,0]
 def spatial_states(self,edge,fraction):
  slot=np.minimum((fraction*self.bins[edge]).astype(np.int64),self.bins[edge]-1)
  return self.state_offsets[edge]+slot
@njit(cache=True,parallel=True)
def fitted_pair_distances(edge,fraction,continues,u,v,length,D,outj,outl,inj,inl,chain,pos,prefix):
 result=np.full(len(edge)-1,np.inf)
 for i in prange(len(result)):
  if continues[i]:
   result[i]=edge_distance(edge[i],fraction[i],edge[i+1],fraction[i+1],u,v,length,D,outj,outl,inj,inl,chain,pos,prefix)
 return result
