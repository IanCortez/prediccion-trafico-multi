from common import *
from consultas_preferidas import learned_features,NAMES
from map_matching import node_connections
from ruteo_compacto import compressed_csr
from numba import set_num_threads
import argparse
p=argparse.ArgumentParser();p.add_argument('--verify-only',action='store_true');a=p.parse_args();started=time.time();set_num_threads(4)
folder=DATA/'preferencias_giro_2014';gate=json.loads((BASE/'instrumentos/preferencias_giro_oof.json').read_text());assert gate['plan_decoder'];raw=dict(np.load(DATA/'mapa_dirigido_2014.npz'));comp=dict(np.load(DATA/'grafo_compacto_2014.npz'));s=dict(np.load(folder/'estructura.npz'));q=dict(np.load(DATA/'consultas_proyectadas_2014.npz'));D=np.load(DATA/'distancias_2014/D.npy',mmap_mode='r');conn=node_connections(comp);graph,chosen=compressed_csr(comp,'length');routing=(comp['chain_offsets'],comp['flat_incoming_arc'],graph.indptr,graph.indices,chosen,comp['chain_length']);profiles=DATA/'perfiles_mapmatched_2014'
args=(raw['u'],raw['v'],raw['length'],raw['seconds'],raw['xy'],D,conn,routing,comp['to_junction'],comp['junction_raw'],comp['chain_dst'],comp['chain_length'],comp['prefix_length'],s['outptr'],s['outchains'],s['transition_offsets'],s['arc_chain'],s['heading_in'],s['heading_out'],np.load(folder/'local.npy',mmap_mode='r'),np.load(folder/'global.npy'),np.load(profiles/'time_s.npy',mmap_mode='r'),np.load(profiles/'state_support.npy',mmap_mode='r'),np.load(profiles/'link_support.npy',mmap_mode='r'))
parts=[dict(np.load(DATA/(split+'_x.npz'))) for split in ('train','validation','test')];bounds=np.r_[0,np.cumsum([len(d['row']) for d in parts])];od_m=np.concatenate([d['od_m'] for d in parts]);assert np.array_equal(q['row'],np.concatenate([d['row'] for d in parts]))
ix=np.random.default_rng(1729).choice(len(q['row']),2000,replace=False) if a.verify_only else np.arange(len(q['row']))
X,errors=learned_features(q['origin'][ix],q['destination'][ix],od_m[ix],q['o_edge'][ix],q['o_fraction'][ix],q['o_error'][ix],q['d_edge'][ix],q['d_fraction'][ix],q['d_error'][ix],q['profile'][ix],*args);maximum=errors.max(0);assert maximum[0]==0 and maximum[1]<.001 and maximum[2]<.01 and maximum[3]<.02,maximum;assert np.isfinite(X).all();assert np.all((X[:,23:25]>=0)&(X[:,23:25]<=1.0001))
report=dict(verify_only=a.verify_only,n=len(ix),max_continuity_violations=float(maximum[0]),max_excess_edge_part_m=float(maximum[1]),max_endpoint_error_m=float(maximum[2]),max_polyline_length_error_m=float(maximum[3]),fallback_fraction=float(X[:,27].mean()),available_fraction=float(X[:,20].mean()),junctions_quantiles=np.quantile(X[:,30]*100,[0,.5,.9,.99,1]),test_labels_read=False,source_preferences_sha256=sha(folder/'manifiesto.json'),source_profiles_sha256=sha(profiles/'manifiesto.json'),route='Greedy whole-trip-week-excluded turn policy with fixed Dirichlet20 smoothing, network500m prior, loop and detour controls; shortest remaining route fallback. OD only.',seconds=time.time()-started)
if a.verify_only:write_json(BASE/'instrumentos/ruta_preferida_geometria.json',report)
else:
    hashes={}
    for i,split in enumerate(('train','validation','test')):
        lo,hi=bounds[i:i+2];path=DATA/(split+'_ruta_preferida.npz');assert not path.exists();np.savez_compressed(path,row=q['row'][lo:hi],X=X[lo:hi],names=np.array(NAMES));hashes[path.name]=sha(path)
    report['outputs']=hashes;report['code_hashes']={name:sha(BASE/name) for name in ('ruta_preferida.py','consultas_preferidas.py','preparar_ruta_preferida.py')};write_json(DATA/'ruta_preferida_manifiesto.json',report)
print(json.dumps(report,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)),flush=True)
