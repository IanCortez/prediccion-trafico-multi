from common import *
from map_matching import project_candidates
from scipy.spatial import cKDTree
from numba import set_num_threads
set_num_threads(4);raw=dict(np.load(DATA/'mapa_dirigido_2014.npz'));z=np.load(DATA/'trayectorias_train.npz');xy=z['xy'];points=xy[np.random.default_rng(175).choice(len(xy),2000,replace=False)].astype(float)*1000
count=np.maximum(1,np.ceil(raw['length']/25).astype(int));sample_edge=np.repeat(np.arange(len(count),dtype=np.int32),count);local=np.arange(len(sample_edge))-np.repeat(np.cumsum(count)-count,count);t=(local+.5)/count[sample_edge];A=raw['xy'][raw['u'][sample_edge]];B=raw['xy'][raw['v'][sample_edge]];tree=cKDTree(A+(B-A)*t[:,None]);_,near=tree.query(points,k=32,workers=4)
edges,fraction,error=project_candidates(points,near,sample_edge,raw['xy'],raw['u'],raw['v']);assert np.all(edges>=0);max_error=0.;max_frac=0.
for i in range(len(points)):
    ids=np.unique(sample_edge[near[i]]);a=raw['xy'][raw['u'][ids]];delta=raw['xy'][raw['v'][ids]]-a;f=np.clip(np.sum((points[i]-a)*delta,axis=1)/np.sum(delta*delta,axis=1),0,1);e=np.linalg.norm(points[i]-a-f[:,None]*delta,axis=1);max_error=max(max_error,abs(float(error[i,0])-float(e.min())))
    for slot in range(4):
        j=np.flatnonzero(ids==edges[i,slot])[0];max_frac=max(max_frac,abs(f[j]-fraction[i,slot]));assert abs(e[j]-error[i,slot])<.005
assert max_error<.005 and max_frac<1e-6
write_json(BASE/'instrumentos/proyeccion_gps_v2.json',dict(passed=True,n=2000,independent_numpy_projection=True,max_residual_error_m=max_error,max_fraction_error=max_frac,parallel_point_projection=True,test_labels_read=False,partial_v1_preserved=True))
print(dict(max_error_m=max_error,max_fraction_error=max_frac),flush=True)
