from common import *
from features import ORIGIN
from map_matching import project_candidates,node_connections
from ruteo_proyectado import free_connections
from consultas_mapmatched import projected_route_features
from ruteo_compacto import compressed_csr
from scipy.spatial import cKDTree
from numba import set_num_threads
import pandas as pd
started=time.time();set_num_threads(4);raw=dict(np.load(DATA/'mapa_dirigido_2014.npz'));comp=dict(np.load(DATA/'grafo_compacto_2014.npz'));profiles=DATA/'perfiles_mapmatched_2014';pm=json.loads((profiles/'manifiesto.json').read_text());free_info=json.loads((DATA/'distancias_2014_free/manifiesto.json').read_text());assert free_info['raw_graph_sha256']==sha(DATA/'mapa_dirigido_2014.npz')
parts=[dict(np.load(DATA/(s+'_x.npz'))) for s in ('train','validation','test')];d={key:np.concatenate([p[key] for p in parts]) for key in parts[0]};N=len(d['row']);bounds=np.r_[0,np.cumsum([len(p['row']) for p in parts])]
cache=DATA/'consultas_proyectadas_2014.npz'
if cache.exists():
    q=dict(np.load(cache));assert np.array_equal(q['row'],d['row'])
else:
    def metric(tag):return np.column_stack([(d[tag+'_lon']-ORIGIN[0])*111195*np.cos(np.deg2rad(ORIGIN[1])),(d[tag+'_lat']-ORIGIN[1])*111195])
    origin=metric('o');destination=metric('d');count=np.maximum(1,np.ceil(raw['length']/25).astype(int));sample_edge=np.repeat(np.arange(len(count),dtype=np.int32),count);index=np.arange(len(sample_edge))-np.repeat(np.cumsum(count)-count,count);t=(index+.5)/count[sample_edge];A=raw['xy'][raw['u'][sample_edge]];B=raw['xy'][raw['v'][sample_edge]];tree=cKDTree(A+(B-A)*t[:,None]);q=dict(row=d['row'],origin=origin,destination=destination)
    for tag,xy in (('o',origin),('d',destination)):
        edges=np.empty((N,4),np.int32);fraction=np.empty((N,4),np.float32);error=np.empty_like(fraction)
        for first in range(0,N,100000):
            last=min(first+100000,N);distance,near=tree.query(xy[first:last],k=32,workers=4);del distance;e,f,r=project_candidates(xy[first:last],near,sample_edge,raw['xy'],raw['u'],raw['v']);assert np.all(e>=0);edges[first:last]=e;fraction[first:last]=f;error[first:last]=r
        q[tag+'_edge']=edges;q[tag+'_fraction']=fraction;q[tag+'_error']=error
    local=pd.to_datetime(d['timestamp'],unit='s',utc=True).tz_convert('Europe/Lisbon');how=np.asarray(local.dayofweek,dtype=np.int64)*24+np.asarray(local.hour,dtype=np.int64);fold=np.full(N,5,np.int64);fold[:bounds[1]]=(d['timestamp'][:bounds[1]]//(7*86400))%5;q['profile']=(fold*168+how).astype(np.int32);np.savez_compressed(cache,**q)
assert np.all(q['profile'][:bounds[1]]<840) and np.all(q['profile'][bounds[1]:]>=840)
profile_time=np.load(profiles/'time_s.npy',mmap_mode='r');state_support=np.load(profiles/'state_support.npy',mmap_mode='r');link_support=np.load(profiles/'link_support.npy',mmap_mode='r');static_parts=[];movement_parts=[];static_names=[];movement_names=[];checks={}
for kind in ('length','free'):
    D=np.load(DATA/('distancias_2014'+('_free' if kind=='free' else ''))/'D.npy',mmap_mode='r');conn=free_connections(comp) if kind=='free' else node_connections(comp);graph,chosen=compressed_csr(comp,kind);routing=(comp['chain_offsets'],comp['flat_incoming_arc'],graph.indptr,graph.indices,chosen,comp['chain_free'] if kind=='free' else comp['chain_length']);arc_cost=raw['seconds'] if kind=='free' else raw['length']
    X,M,error=projected_route_features(q['origin'],q['destination'],d['od_m'],q['o_edge'],q['o_fraction'],q['o_error'],q['d_edge'],q['d_fraction'],q['d_error'],q['profile'],raw['u'],raw['v'],raw['length'],raw['seconds'],raw['xy'],arc_cost,D,conn,routing,profile_time,state_support,link_support,kind=='free')
    maxima=error.max(0);assert maxima[0]<.1 and maxima[1]<.01 and maxima[2]<.02,(kind,maxima)
    assert np.isfinite(X).all() and np.isfinite(M).all();assert np.all((M[:,1:3]>=0)&(M[:,1:3]<=1.0001));static_parts.append(X);movement_parts.append(M)
    static_names += ['projected_'+kind+'_shape_'+str(i) for i in range(14)]+['projected_'+kind+'_'+v for v in ('polyline_km','circuity_log','turn_abs_pi','turn_signed_pi','sharp_turn_count','max_turn_pi','available','freeflow_600s','o_projection_km','d_projection_km','attachment_gap','reachable_fraction')]
    movement_names += ['mm_'+kind+'_'+v for v in ('gps_time_600s','state_support','link_support','congestion_log','pace_log')];checks[kind]=dict(max_planning_cost_error=float(maxima[0]),cost_units='s' if kind=='free' else 'm',max_endpoint_error_m=float(maxima[1]),max_polyline_length_error_m=float(maxima[2]),available_fraction=float(X[:,20].mean()));print(dict(kind=kind,checks=checks[kind],seconds=time.time()-started),flush=True)
all_features={'rutas_proyectadas2014':(np.column_stack(static_parts),static_names),'gps_mapmatched':(np.column_stack(movement_parts),movement_names)};hashes={}
for component,(X,names) in all_features.items():
    assert X.shape[1]==len(names)
    for i,split in enumerate(('train','validation','test')):
        lo,hi=bounds[i:i+2];path=DATA/(split+'_'+component+'.npz');assert not path.exists(),path;np.savez_compressed(path,row=d['row'][lo:hi],X=X[lo:hi],names=np.array(names));hashes[path.name]=sha(path)
write_json(DATA/'mapmatched_features_manifiesto.json',dict(n_queries=N,static_feature_count=len(static_names),movement_feature_count=len(movement_names),query_cache_sha256=sha(cache),profiles_sha256=sha(profiles/'manifiesto.json'),outputs=hashes,checks=checks,training_history='Excludes query UTC-week fold at every hierarchy level',heldout_history='Full training GPS only',route_selection='Four OD edge projections; minimize static length plus 3*connector distance, or free-flow seconds plus connector at10kmh. Movement profiles do not choose these routes.',inference='OD and departure only. No evaluated-query observed route, heading, point count or duration.',test_labels_read=False,code_hashes={name:sha(BASE/name) for name in ('ruteo_proyectado.py','consultas_mapmatched.py','preparar_consultas_mapmatched.py','map_matching.py')},seconds=time.time()-started))
print(dict(stage='done',seconds=time.time()-started),flush=True)
