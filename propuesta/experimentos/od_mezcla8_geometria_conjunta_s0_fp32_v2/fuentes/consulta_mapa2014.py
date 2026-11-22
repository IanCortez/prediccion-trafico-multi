from common import *
from functools import lru_cache
from features import ORIGIN
from map_matching import project_candidates,node_connections
from ruteo_proyectado import free_connections
from ruteo_compacto import compressed_csr
from consultas_mapmatched import projected_route_features
from scipy.spatial import cKDTree
from numba import set_num_threads
import pandas as pd

@lru_cache(maxsize=1)
def resources():
    raw=dict(np.load(DATA/'mapa_dirigido_2014.npz'));comp=dict(np.load(DATA/'grafo_compacto_2014.npz'));count=np.maximum(1,np.ceil(raw['length']/25).astype(int));sample_edge=np.repeat(np.arange(len(count),dtype=np.int32),count);index=np.arange(len(sample_edge))-np.repeat(np.cumsum(count)-count,count);t=(index+.5)/count[sample_edge];A=raw['xy'][raw['u'][sample_edge]];B=raw['xy'][raw['v'][sample_edge]];tree=cKDTree(A+(B-A)*t[:,None]);routes={}
    for kind in ('length','free'):
        D=np.load(DATA/('distancias_2014'+('_free' if kind=='free' else ''))/'D.npy',mmap_mode='r');conn=free_connections(comp) if kind=='free' else node_connections(comp);graph,chosen=compressed_csr(comp,kind);routing=(comp['chain_offsets'],comp['flat_incoming_arc'],graph.indptr,graph.indices,chosen,comp['chain_free'] if kind=='free' else comp['chain_length']);routes[kind]=(D,conn,routing)
    folder=DATA/'perfiles_mapmatched_2014';profiles=tuple(np.load(folder/name,mmap_mode='r') for name in ('time_s.npy','state_support.npy','link_support.npy'))
    return raw,comp,sample_edge,tree,routes,profiles

def build_2014(d,cfg):
    manifest=json.loads((DATA/'manifiesto.json').read_text());assert np.all(d['timestamp']>=manifest['validation']['start_timestamp']),'Raw-query adapter uses full training history; departures must follow the frozen training window.'
    set_num_threads(4);raw,comp,sample_edge,tree,routes,profiles=resources()
    def metric(tag):return np.column_stack([(d[tag+'_lon']-ORIGIN[0])*111195*np.cos(np.deg2rad(ORIGIN[1])),(d[tag+'_lat']-ORIGIN[1])*111195])
    origin=metric('o');destination=metric('d');projected=[]
    for xy in (origin,destination):
        _,near=tree.query(xy,k=32,workers=4);projected.extend(project_candidates(xy,near,sample_edge,raw['xy'],raw['u'],raw['v']))
    oe,of,od,de,df,dd=projected;local=pd.to_datetime(d['timestamp'],unit='s',utc=True).tz_convert('Europe/Lisbon');profile=(840+np.asarray(local.dayofweek)*24+np.asarray(local.hour)).astype(np.int32);static_parts=[];movement_parts=[];static_names=[];movement_names=[]
    for kind in ('length','free'):
        D,conn,routing=routes[kind];X,M,error=projected_route_features(origin,destination,d['od_m'],oe,of,od,de,df,dd,profile,raw['u'],raw['v'],raw['length'],raw['seconds'],raw['xy'],raw['seconds'] if kind=='free' else raw['length'],D,conn,routing,*profiles,kind=='free');assert error.max()<.1;static_parts.append(X);movement_parts.append(M)
        static_names+=['projected_'+kind+'_shape_'+str(i) for i in range(14)]+['projected_'+kind+'_'+v for v in ('polyline_km','circuity_log','turn_abs_pi','turn_signed_pi','sharp_turn_count','max_turn_pi','available','freeflow_600s','o_projection_km','d_projection_km','attachment_gap','reachable_fraction')];movement_names+=['mm_'+kind+'_'+v for v in ('gps_time_600s','state_support','link_support','congestion_log','pace_log')]
    parts=[];names=[]
    if cfg.get('projected_routes',False) or cfg.get('mapmatched',False):parts+=static_parts;names+=static_names
    if cfg.get('mapmatched',False):parts+=movement_parts;names+=movement_names
    if cfg.get('learned_route',False):
        from consultas_preferidas import learned_features,NAMES
        folder=DATA/'preferencias_giro_2014';s=dict(np.load(folder/'estructura.npz'));D,conn,routing=routes['length'];X,error=learned_features(origin,destination,d['od_m'],oe,of,od,de,df,dd,profile,raw['u'],raw['v'],raw['length'],raw['seconds'],raw['xy'],D,conn,routing,comp['to_junction'],comp['junction_raw'],comp['chain_dst'],comp['chain_length'],comp['prefix_length'],s['outptr'],s['outchains'],s['transition_offsets'],s['arc_chain'],s['heading_in'],s['heading_out'],np.load(folder/'local.npy',mmap_mode='r'),np.load(folder/'global.npy'),*profiles);assert error.max()<.1;parts.append(X);names+=NAMES
    if cfg.get('road_attributes',False):
        from atributos_ruteados import attribute_features,FIELDS
        attrs=dict(np.load(DATA/'atributos_mapa2014.npz'))
        for kind in ('length','free'):
            D,conn,routing=routes[kind];X,error=attribute_features(oe,of,od,de,df,dd,raw['u'],raw['v'],raw['length'],raw['seconds'],raw['seconds'] if kind=='free' else raw['length'],D,conn,routing,attrs['road_class'],attrs['lanes'],attrs['flags'],attrs['node_control'],attrs['degree'],kind=='free');assert error.max()<.1;parts.append(X);names+=['attrs_'+kind+'_'+n for n in FIELDS]
    return parts,names
