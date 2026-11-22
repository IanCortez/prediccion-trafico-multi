from common import *
from features import *
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
raw=dict(np.load(DATA/'mapa_dirigido.npz'));comp=dict(np.load(DATA/'grafo_compacto.npz'));q=dict(np.load(DATA/'consultas_ruteo_compacto.npz'));arc=dict(np.load(DATA/'arcos_tiempos_gps.npz'))
X=np.concatenate([np.load(DATA/(s+'_rutas_dinamicas.npz'))['X'] for s in ('train','validation','test')]);errors=[]
for profile in (0,12,24,36,48,60,63,71):
    weight=arc['time_s'][profile].astype(float);n=len(raw['xy']);pair=raw['u'].astype(np.int64)*n+raw['v'];order=np.lexsort((weight,pair));chosen=order[np.r_[True,np.diff(pair[order])!=0]]
    graph=csr_matrix((weight[chosen],(raw['u'][chosen],raw['v'][chosen])),shape=(n,n))
    ids=np.flatnonzero((q['profile_id']==profile)&(X[:,20]>0));ids=np.random.default_rng(623+profile).choice(ids,4,replace=False)
    for i in ids:
        def snapped(tag):
            chain=q[tag+'_chain'][i,0]
            return comp['flat_nodes'][q[tag+'_pos'][i,0]] if chain>=0 else comp['junction_raw'][q[tag+'_junction'][i,0]]
        origin=snapped('o');dest=snapped('d');distance=dijkstra(graph,directed=True,indices=[origin],return_predecessors=False)[0,dest]
        exact=distance+(q['o_snap'][i]+q['d_snap'][i])/(20/3.6);errors.append(abs(float(exact-X[i,21]*600)))
assert max(errors)<.1,max(errors)
d=np.load(DATA/'validation_x.npz');geometry=np.load(DATA/'geometria_validation.npz')['X'][:,1:15].reshape(-1,7,2);static=np.load(DATA/'validation_geometria_rutas.npz')['X'];dynamic=np.load(DATA/'validation_rutas_dinamicas.npz')['X'];coverage={};differences=[]
for name,featureset,start,available in (('shortest',static,0,20),('freeflow',static,21,41),('historical_gps',dynamic,0,20)):
    candidates=featureset[:,start:start+14].reshape(-1,7,2);error=np.linalg.norm(candidates-geometry,axis=-1).mean(1);error[featureset[:,available]<=0]=np.inf;differences.append(error)
    coverage[name]=dict(median_shape_error_m=float(np.median(error)*1000),fraction_within200m=float(np.mean(error<=.2)),fraction_within500m=float(np.mean(error<=.5)))
best=np.minimum.reduce(differences);coverage['best_of_3_privileged_choice']=dict(median_shape_error_m=float(np.median(best)*1000),fraction_within200m=float(np.mean(best<=.2)),fraction_within500m=float(np.mean(best<=.5)))
write_json(BASE/'instrumentos/rutas_dinamicas.json',dict(passed=True,independent_raw_graph_queries=len(errors),max_cost_difference_s=max(errors),coverage_diagnostic_validation=coverage,test_labels_read=False,caveat='Candidate quality uses observed validation geometry for diagnosis only, never model inputs or cohort filtering. Best of 3 is a privileged selector, not an OD prediction.'))
print(dict(max_cost_difference_s=max(errors),coverage=coverage),flush=True)
