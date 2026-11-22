from common import *
from features import *
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
z=np.load(DATA/"mapa_dirigido.npz");n=len(z["xy"])
checks={}
checks["finite_positive_edges"]=bool(np.isfinite(z["length"]).all() and (z["length"]>0).all() and (z["seconds"]>0).all())
checks["unique_arcs"]=len(np.unique(z["u"].astype(np.int64)*n+z["v"]))==len(z["u"])
pairs=set(zip(z["u"].tolist(),z["v"].tolist()))
oneway=sum((v,u) not in pairs for u,v in pairs)
checks["directions_retained"]=oneway>0
active=np.unique(np.concatenate([z["u"],z["v"]]));tree=cKDTree(z["xy"][active])
d=dict(np.load(DATA/"validation_x.npz"));r=np.load(DATA/"validation_rutas.npz")
checks["row_alignment"]=np.array_equal(d["row"],r["row"])
checks["route_features_finite"]=bool(np.isfinite(r["X"]).all())
g=csr_matrix((z["length"],(z["u"],z["v"])),shape=(n,n))
gt=csr_matrix((z["seconds"],(z["u"],z["v"])),shape=(n,n))
errs=[];terrs=[]
for i in np.random.default_rng(91).choice(len(d["row"]),12,replace=False):
    def xy(tag):
        return np.array([(d[tag+"_lon"][i]-ORIGIN[0])*111195*np.cos(np.deg2rad(ORIGIN[1])),(d[tag+"_lat"][i]-ORIGIN[1])*111195])
    os,oi=tree.query(xy("o"));ds,di=tree.query(xy("d"));oi,di=active[oi],active[di]
    length=dijkstra(g,directed=True,indices=oi)[di]+os+ds
    sec=dijkstra(gt,directed=True,indices=oi)[di]+(os+ds)/(20/3.6)
    if r["X"][i,-1] and np.isfinite(length):
        errs.append(abs(length-r["X"][i,0]*1000));terrs.append(abs(sec-r["X"][i,1]*600))
checks["independent_route_recalculation"]=bool(max(errs)<.01 and max(terrs)<.01)
aug=np.load(DATA/"subtrayectos.npz");train=np.load(DATA/"train_x.npz");test=np.load(DATA/"test_x.npz")
parents=np.unique(aug["parent_row"])
checks["augmentation_train_only"]=bool(np.isin(parents,train["row"]).all() and not np.isin(parents,d["row"]).any() and not np.isin(parents,test["row"]).any())
checks["augmentation_arrival_embargo"]=bool((aug["timestamp"]+aug["duration"]).max()<d["timestamp"].min())
out=dict(checks=checks,passed=all(checks.values()),oneway_arcs=oneway,route_recalculation_max_error_m=max(errs),route_recalculation_max_error_s=max(terrs),test_labels_read=False)
write_json(BASE/"instrumentos/mapa_y_preentrenamiento.json",out);assert out["passed"],out;print(out,flush=True)

