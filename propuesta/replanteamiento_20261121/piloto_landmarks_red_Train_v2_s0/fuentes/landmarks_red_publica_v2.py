"""Public directed-network landmark geometry; no routegeneration, querytrajectory or labels."""
from base_replanteo import *
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
def xy_query(d):
 sx=111195*np.cos(np.deg2rad(41.16))
 return np.column_stack([(d['o_lon']+8.62)*sx,(d['o_lat']-41.16)*111195,
                         (d['d_lon']+8.62)*sx,(d['d_lat']-41.16)*111195])
class PublicLandmarks:
 def __init__(self):
  self.comp=dict(np.load(DATA/'grafo_compacto_2014.npz'));self.raw=dict(np.load(DATA/'mapa_dirigido_2014.npz'))
  c=self.comp;N=len(c['junction_raw']);xy=self.raw['xy'][c['junction_raw']]
  graph=csr_matrix((np.ones(len(c['chain_src'])),(c['chain_src'],c['chain_dst'])),shape=(N,N))
  _,label=connected_components(graph,directed=True,connection='strong')
  main=np.argmax(np.bincount(label));ll=self.raw['lonlat'][c['junction_raw']]
  city=(ll[:,0]>=-8.6887)&(ll[:,0]<=-8.5557)&(ll[:,1]>=41.1405)&(ll[:,1]<=41.1865)&(label==main)
  pool=np.flatnonzero(city);assert len(pool)>=32
  point=xy[pool];center=point.mean(0);selected=[int(np.argmin(np.sum((point-center)**2,1)))];distance=np.full(len(pool),np.inf)
  for k in range(1,32):
   distance=np.minimum(distance,np.sum((point-point[selected[-1]])**2,1));distance[selected]=-1
   selected.append(int(np.argmax(distance)))
  self.landmarks=pool[np.asarray(selected)];self.landmark_xy=xy[self.landmarks]
  D=np.load(DATA/'distancias_2014/D.npy',mmap_mode='r')
  self.out=np.ascontiguousarray(D[:,self.landmarks]);self.into=np.ascontiguousarray(D[self.landmarks,:].T)
  self.D=D;self._projection_resources=None
 def features(self,d,projection):
  q=xy_query(d);c=self.comp;features=[];controls=[];flags=[]
  for e in range(2):
   chain=projection['chain'][:,e].astype(np.int64);f=projection['fraction'][:,e].astype(float).clip(0,1)
   error=projection['error_m'][:,e].astype(float)
   assert chain.min()>=0 and chain.max()<len(c['chain_src']) and np.isfinite(error).all()
   src=c['chain_src'][chain];dst=c['chain_dst'][chain];length=c['chain_length'][chain]
   outgoing=(1-f)[...,None]*length[...,None]+self.out[dst]
   outgoing=np.where((f==0)[...,None],np.minimum(outgoing,self.out[src]),outgoing)
   incoming=f[...,None]*length[...,None]+self.into[src]
   incoming=np.where((f==1)[...,None],np.minimum(incoming,self.into[dst]),incoming)
   euclidean=np.linalg.norm(q[:,None,e*2:e*2+2]-self.landmark_xy[None],axis=2)
   for values in [outgoing,incoming]:
    values=values+error[...,None];valid=np.isfinite(values)
    logweight=np.broadcast_to((-.5*(error/30)**2)[...,None],values.shape)
    logweight=np.where(valid,logweight,-1e30)
    weight=np.exp(logweight-logweight.max(1,keepdims=True))*valid
    denominator=weight.sum(1);have=denominator>0
    result=np.sum(weight*np.where(valid,values,0),axis=1)/np.maximum(denominator,1e-30)
    result=np.where(have,result,50000.).clip(0,50000.)
    features.append(np.log1p(result/1000));controls.append(np.log1p(euclidean/1000));flags.append(have.all(1))
  # Endpoint/order: Oout,Oin,Dout,Din, each32; four availabilityflags.
  topo=np.column_stack(features+flags).astype(np.float32)
  euclid=np.column_stack(controls+[np.ones(len(q)) for _ in range(4)]).astype(np.float32)
  assert topo.shape==euclid.shape==(len(q),132) and np.isfinite(topo).all()
  return topo,euclid
 def raw_projection(self,items):
  from consulta_od import od_records
  from consulta_mapa2014 import resources,ORIGIN
  from map_matching import project_candidates
  from numba import set_num_threads
  d=od_records(items);set_num_threads(2)
  if self._projection_resources is None:
   raw,c,sample,tree,_,_=resources()
   flat=c['flat_incoming_arc'];valid=flat>=0;pc=np.repeat(np.arange(len(c['chain_src']),dtype=np.int32),np.diff(c['chain_offsets']))
   edge_chain=np.full(len(raw['u']),-1,np.int32);edge_chain[flat[valid]]=pc[valid]
   edge_position=np.empty(len(raw['u']),np.int32);edge_position[flat[valid]]=np.flatnonzero(valid)
   edge_prefix=c['prefix_length'][edge_position-1]
   self._projection_resources=raw,c,sample,tree,edge_chain,edge_prefix
  raw,c,sample,tree,edge_chain,edge_prefix=self._projection_resources
  ll=np.stack([np.column_stack([d['o_lon'],d['o_lat']]),np.column_stack([d['d_lon'],d['d_lat']])],1).reshape(-1,2)
  points=np.column_stack([(ll[:,0]-ORIGIN[0])*111195*np.cos(np.deg2rad(ORIGIN[1])),(ll[:,1]-ORIGIN[1])*111195])
  _,near=tree.query(points,k=32,workers=2)
  edge,frac,error=project_candidates(points,near,sample,raw['xy'],raw['u'],raw['v'])
  chain=edge_chain[edge].reshape(-1,2,4);pos=(edge_prefix[edge]+frac*raw['length'][edge]).reshape(-1,2,4)
  length=c['chain_length'][chain];fraction=np.divide(pos,length,out=np.zeros_like(pos),where=length>0).astype(np.float16)
  return d,dict(chain=chain.astype(np.uint16),fraction=fraction,error_m=error.reshape(-1,2,4).astype(np.float16))
