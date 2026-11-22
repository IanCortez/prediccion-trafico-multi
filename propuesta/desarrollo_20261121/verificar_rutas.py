"""Independent graph optimum and batch-invariance checks for global OD routing."""
from piloto_transferencia import ROOT,BASE,OUT,write,sha
from rutas_globales import astar
from generar_rutas import route_batch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from numba import set_num_threads
import numpy as np,pandas as pd,time
start=time.time();set_num_threads(2);raw=dict(np.load(BASE/'datos/mapa_dirigido_2014.npz'));xy=raw['xy'];order=np.argsort(raw['u'],kind='stable').astype(np.int32);ptr=np.r_[0,np.cumsum(np.bincount(raw['u'],minlength=len(xy)))].astype(np.int64);dest=raw['v'][order];times=np.load(BASE/'datos/perfiles_mapmatched_2014/time_s.npy',mmap_mode='r');support=np.load(BASE/'datos/perfiles_mapmatched_2014/link_support.npy',mmap_mode='r');d=dict(np.load(BASE/'datos/validation_x.npz'));ix=np.load(OUT/'rutas_globales/validation_diagnostico.npz')['indices'][:16];origin=np.column_stack([(d['o_lon'][ix]+8.62)*111195*np.cos(np.deg2rad(41.16)),(d['o_lat'][ix]-41.16)*111195]);end=np.column_stack([(d['d_lon'][ix]+8.62)*111195*np.cos(np.deg2rad(41.16)),(d['d_lat'][ix]-41.16)*111195]);tree=cKDTree(xy);_,src=tree.query(origin);_,tar=tree.query(end);dt=pd.to_datetime(d['timestamp'][ix],unit='s',utc=True).tz_convert('Europe/Lisbon');profiles=(840+np.asarray(dt.dayofweek)*24+np.asarray(dt.hour)).astype(np.int32)
key=raw['u'].astype(np.int64)*len(xy)+raw['v'];ko=np.argsort(key);keys=key[ko];starts=np.r_[0,np.flatnonzero(np.diff(keys))+1];unique=keys[starts];rows=unique//len(xy);cols=unique%len(xy);errors=[];count=0
for s,t,p in zip(src,tar,profiles):
 cost=times[p].astype(np.float64)*(1-.6*support[5].astype(np.float64));minimum=np.minimum.reduceat(cost[ko],starts);graph=csr_matrix((minimum,(rows,cols)),shape=(len(xy),len(xy)));expected=dijkstra(graph,directed=True,indices=int(s))[t];nodes,edges,expanded=astar(s,t,xy,ptr,dest,order,raw['length'],raw['seconds'],times,support,p,2)
 if np.isfinite(expected):
  assert len(nodes) and nodes[0]==s and nodes[-1]==t;assert np.array_equal(raw['u'][edges],nodes[:-1]) and np.array_equal(raw['v'][edges],nodes[1:]);error=abs(cost[edges].sum()-expected);assert error<.001,(error,s,t,p);errors.append(error);count+=1
 else:assert not len(nodes)
a=route_batch(origin,end,src,tar,profiles,xy,ptr,dest,order,raw['length'],raw['seconds'],times,support);b=route_batch(origin[::-1].copy(),end[::-1].copy(),src[::-1].copy(),tar[::-1].copy(),profiles[::-1].copy(),xy,ptr,dest,order,raw['length'],raw['seconds'],times,support)[::-1];assert np.array_equal(a,b)
write(OUT/'verificacion_rutas.json',dict(passed=True,independent_shortest_path_reference='SciPy Dijkstra with minimum, not sum, for parallel edges',finite_pairs=count,max_cost_error_s=max(errors),batch_reversal_bit_identical=True,Test_access=False,seconds=time.time()-start,code_sha256=sha(ROOT.__class__(__file__))))
print('Independent route optimum and batch-invariance checks passed.',flush=True)
