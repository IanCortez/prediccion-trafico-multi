from common import *
from map_matching import *
from compresion_vial import compress_graph
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from numba import set_num_threads
set_num_threads(3)
raw=dict(np.load(DATA/'mapa_dirigido_2014.npz'));comp=dict(np.load(DATA/'grafo_compacto_2014.npz'));D=np.load(DATA/'distancias_2014/D.npy',mmap_mode='r');connections=node_connections(comp);rng=np.random.default_rng(745)
source=rng.choice(len(raw['xy']),30,replace=False);destination=rng.choice(len(raw['xy']),70,replace=False);g=csr_matrix((raw['length'],(raw['u'],raw['v'])),shape=(len(raw['xy']),)*2);exact=dijkstra(g,directed=True,indices=source,return_predecessors=False)[:,destination]
calc=np.array([[node_distance(a,b,D,*connections) for b in destination] for a in source]);assert np.array_equal(np.isfinite(exact),np.isfinite(calc));valid=np.isfinite(exact);node_error=float(np.max(np.abs(exact[valid]-calc[valid])));assert node_error<.02,node_error
# Endpoint-constrained edge distance, verified independently on raw nodes.
eas=rng.choice(len(raw['u']),40,replace=False);ebs=rng.choice(len(raw['u']),40,replace=False);endpoints=np.unique(np.r_[raw['u'][eas],raw['v'][eas]]);table=dijkstra(g,directed=True,indices=endpoints,return_predecessors=False);lookup={node:i for i,node in enumerate(endpoints)};edge_errors=[]
for i,(ea,eb) in enumerate(zip(eas,ebs)):
    fa=[0.,.25,.75,1.][i%4];fb=[0.,.4,.8,1.][(i//4)%4];L=raw['length'];u=raw['u'];v=raw['v'];expected=(1-fa)*L[ea]+table[lookup[v[ea]],u[eb]]+fb*L[eb]
    if ea==eb and fb>=fa:expected=min(expected,(fb-fa)*L[ea])
    if fa==0:expected=min(expected,table[lookup[u[ea]],u[eb]]+fb*L[eb])
    if fb==1:expected=min(expected,(1-fa)*L[ea]+table[lookup[v[ea]],v[eb]])
    if fa==0 and fb==1:expected=min(expected,table[lookup[u[ea]],v[eb]])
    actual=edge_distance(ea,fa,eb,fb,u,v,L,D,*connections);assert np.isfinite(actual)==np.isfinite(expected)
    if np.isfinite(expected):edge_errors.append(abs(expected-actual))
assert max(edge_errors)<.02,max(edge_errors)
# Directed-state recovery on a bidirectional 300m road, opposite travel directions.
toy=dict(xy=np.array([[0.,0.],[300.,0.]]),u=np.array([0,1]),v=np.array([1,0]),length=np.array([300.,300.]),seconds=np.array([54.,54.]));c=compress_graph(toy);conn=node_connections(c);td=np.array([[0.,300.],[300.,0.]],np.float32)
points=np.array([[50.,0],[100.,0],[150.,0],[250.,0],[200.,0],[150.,0]]);edges=np.tile(np.array([0,1,0,1],np.int32),(6,1));f=points[:,0]/300;fractions=np.column_stack([f,1-f,f,1-f]).astype(np.float32);errors=np.zeros((6,4),np.float32)
matched,_,_,resets=viterbi_match(points,np.array([0,3,6]),edges,fractions,errors,toy['u'],toy['v'],toy['length'],td,*conn);assert np.array_equal(matched,np.array([0,0,0,1,1,1]));assert resets.sum()==0
write_json(BASE/'instrumentos/map_matching_contrato.json',dict(passed=True,raw_node_queries=int(exact.size),max_node_distance_error_m=node_error,constrained_edge_queries=len(eas),max_edge_distance_error_m=max(edge_errors),directed_state_recovery=True,no_travel_time_label_argument=True,test_labels_read=False,caveat='Algorithmic contracts, not measured map-matching accuracy on unlabeled real GPS.'))
print(dict(node_error_m=node_error,edge_error_m=max(edge_errors),toy_directed_recovery=True),flush=True)
