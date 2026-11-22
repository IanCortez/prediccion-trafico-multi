from common import *
from features import *
from ruteo_compacto import *
from geometria_ruteada import sample_routes
from scipy.sparse.csgraph import dijkstra
import math

def od_records(items):
    columns=('o_lon','o_lat','d_lon','d_lat','timestamp')
    assert len(items)>0
    assert all(set(item)==set(columns) for item in items),'Accepts only the five OD/departure fields'
    d={key:np.array([item[key] for item in items],dtype=np.int64 if key=='timestamp' else np.float64) for key in columns}
    assert all(np.isfinite(value).all() for value in d.values())
    for tag in ('o','d'):
        assert np.all((d[tag+'_lat']>=41.1405)&(d[tag+'_lat']<=41.1865)&(d[tag+'_lon']>=-8.6887)&(d[tag+'_lon']<=-8.5557)),'Query outside frozen Porto area'
    distances=[]
    for item in items:
        lat1,lat2=map(math.radians,(item['o_lat'],item['d_lat']));a=math.sin((lat2-lat1)/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin(math.radians(item['d_lon']-item['o_lon'])/2)**2
        distances.append(6371000*2*math.asin(min(1,math.sqrt(a))))
    d['od_m']=np.array(distances,np.float32);assert np.all(d['od_m']>=200)
    d.update(row=np.arange(len(items),dtype=np.int32),call=np.full(len(items),'C'),exposed=np.zeros(len(items),bool))
    return d

def build_od_features(d,cfg):
    assert not cfg.get('archive_maps',[]),'Annual map raw-query adapter pending separate certification'
    assert not any(cfg.get(key,False) for key in ('enriched','gps_routes','history','dynamic_routes','hourly_gps')),'Raw OD builder currently certifies static map/context and optional route shape only'
    F,names=features(d,calendar=cfg.get("calendar",False));parts=[F];raw=dict(np.load(DATA/'mapa_dirigido.npz'));n=len(raw['xy']);active=np.unique(np.r_[raw['u'],raw['v']]);tree=cKDTree(raw['xy'][active])
    def metric(tag):return np.column_stack([(d[tag+'_lon']-ORIGIN[0])*111195*np.cos(np.deg2rad(ORIGIN[1])),(d[tag+'_lat']-ORIGIN[1])*111195])
    os,oi=tree.query(metric('o'));ds,di=tree.query(metric('d'));oi=active[oi];di=active[di];sources,inv=np.unique(oi,return_inverse=True);length=np.empty(len(oi),np.float32);seconds=np.empty_like(length)
    for key,target in (('length',length),('seconds',seconds)):
        graph=csr_matrix((raw[key],(raw['u'],raw['v'])),shape=(n,n))
        for first in range(0,len(sources),32):
            last=min(first+32,len(sources));rows=np.flatnonzero((inv>=first)&(inv<last));dist=dijkstra(graph,directed=True,indices=sources[first:last],return_predecessors=False)
            target[rows]=dist[inv[rows]-first,di[rows]]+(os[rows]+ds[rows])/(20/3.6 if key=='seconds' else 1)
    good=np.isfinite(length)&np.isfinite(seconds)&(os<500)&(ds<500);length=np.where(good,length,d['od_m']).astype(np.float32);seconds=np.where(good,seconds,d['od_m']/(25/3.6)).astype(np.float32)
    routes=np.column_stack([length/1000,seconds/600,np.log(np.maximum(length/d['od_m'],.01)),np.log(np.maximum(seconds*1000/np.maximum(length,1),.01)),os/1000,ds/1000,good.astype(float)]).astype(np.float32)
    parts.append(routes);names+=['network_shortest_km','network_fastest_time_600s','network_circuity_log','network_pace_log','o_snap_km','d_snap_km','route_available']
    grid=dict(np.load(DATA/'contexto_rejilla25m.npz'))
    for tag in ('o','d'):
        xy=metric(tag);ix=np.floor((xy[:,0]-grid['x0'])/grid['gridstep']).astype(int).clip(0,int(grid['nx'])-1);iy=np.floor((xy[:,1]-grid['y0'])/grid['gridstep']).astype(int).clip(0,int(grid['ny'])-1)
        parts.append(grid['features'][ix*grid['ny']+iy]);names += [tag+'_'+name for name in grid['names'].tolist()]
    if cfg.get('map_geometry',False):
        comp=dict(np.load(DATA/'grafo_compacto.npz'));q=query_map(d,raw,comp);sources=np.unique(q['o_junction'][q['o_junction']>=0]);lookup=np.full(len(comp['junction_raw']),-1,np.int32);lookup[sources]=np.arange(len(sources))
        for kind in ('length','free'):
            graph,chosen=compressed_csr(comp,kind);dist,pred=dijkstra(graph,directed=True,indices=sources,return_predecessors=True);_,choice=choose_static(q,dist,lookup,comp,kind)
            shape=sample_routes(raw['xy'],comp['flat_nodes'],comp['chain_offsets'],comp['prefix_length'],comp['junction_raw'],graph.indptr,graph.indices,chosen,pred,lookup,choice,q['o_chain'],q['d_chain'],q['o_pos'],q['d_pos'],q['o_junction'],q['d_junction'],F[:,:2].astype(float)*1000,F[:,2:4].astype(float)*1000)
            valid=(choice>=0)&(q['o_snap']<500)&(q['d_snap']<500);shape[~valid]=0;shape[~valid,14]=F[~valid,6];parts.append(shape)
            names += [kind+'_shape_'+str(i) for i in range(14)]+[kind+'_'+v for v in ('polyline_km','polyline_circuity_log','turn_abs_pi','turn_signed_pi','sharp_turn_count','max_turn_pi','shape_available')]
    if any(cfg.get(key,False) for key in ('projected_routes','mapmatched','learned_route','road_attributes')):
        from consulta_mapa2014 import build_2014
        added,columns=build_2014(d,cfg);parts+=added;names+=columns
    X=np.column_stack(parts).astype(np.float32)
    if cfg.get('weekly_only',False):
        keep=[i for i,name in enumerate(names) if name not in ('month','season_sin','season_cos')];X=X[:,keep];names=[names[i] for i in keep]
    return X,names


