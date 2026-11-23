"""Public shorteststreet signature from five rawfields; no actualqueryroute/length/time accepted."""
from base_replanteo import *
from functools import lru_cache
from numba import set_num_threads
from consulta_mapa2014 import resources,ORIGIN
from proyecciones_valor_vial import project_street_query
from rutas_politica_vial import generate_queries
@lru_cache(maxsize=1)
def static_generator_arguments():
 raw,c,sample,tree,routes,profiles=resources();set_num_threads(2);D,conn,routing=routes['length'];g=dict(np.load(OUT/'banco_valor_vial_s0/grafo_estatico.npz'))
 sx=raw['xy'][c['junction_raw'][c['chain_src']]];dx=raw['xy'][c['junction_raw'][c['chain_dst']]];vec=dx-sx;heading=vec/np.maximum(np.linalg.norm(vec,axis=1,keepdims=True),.001)
 angle=(np.arctan2(vec[:,1],vec[:,0])+2*np.pi)%(2*np.pi);orientation=np.floor(angle*8/(2*np.pi)).astype(np.int32)%8
 speed=c['chain_length']/np.maximum(c['chain_free'],.001);speedbin=np.digitize(speed,[4,7,10,14,20]).astype(np.int32)
 flat=c['flat_incoming_arc'];valid=flat>=0;pc=np.repeat(np.arange(len(c['chain_src']),dtype=np.int32),np.diff(c['chain_offsets']))
 edge_chain=np.full(len(raw['u']),-1,np.int32);edge_chain[flat[valid]]=pc[valid]
 return (g['static'],sx,dx,heading,c['chain_length'],c['chain_free'],orientation,speedbin,g['neighbors'],g['neighbor_valid'],
 D,conn,routing,raw['u'],raw['v'],raw['length'],c['chain_src'],c['chain_dst'],c['chain_offsets'],c['prefix_length'],c['flat_incoming_arc'],edge_chain,
 np.empty((0,1),np.int32),np.empty((0,1),np.float64),np.empty((0,2),np.float64),1.,0.,0)
def public_signature_from_bank(d,bank):
 chord=np.linalg.norm(np.column_stack([(d['d_lon']-d['o_lon'])*111195*np.cos(np.deg2rad(ORIGIN[1])),(d['d_lat']-d['o_lat'])*111195]),axis=1)
 value,bad,tail=generate_queries(bank['chain'].astype(np.int32),bank['fraction'].astype(np.float32),bank['reachable'],chord,np.zeros(len(chord),np.float64),*static_generator_arguments())
 eo=bank['error_m'][:,0].astype(np.float64);ed=bank['error_m'][:,1].astype(np.float64);logits=(-.5*((eo[:,:,None]/30)**2+(ed[:,None,:]/30)**2)).reshape(-1,16)
 logits=np.where(~bad,logits,-1e9);weight=np.exp(logits-logits.max(1,keepdims=True))*(~bad);weight/=np.maximum(weight.sum(1,keepdims=True),1e-20)
 signature=np.einsum('bk,bkf->bf',weight,value).astype(np.float16).astype(np.float32);assert np.isfinite(signature).all()
 return signature,(~bad).any(1)
def raw_public_features(items):
 from consulta_od import od_records,build_od_features
 d=od_records(items);X,names=build_od_features(d,read(BEST/'resultado.json')['config']);keep=np.asarray([not n.startswith('mm_') for n in names]);assert keep.sum()==206
 signature,available=public_signature_from_query(d)
 return np.column_stack([X[:,keep],signature]),[n for n,k in zip(names,keep) if k]+['public_street_signature_'+str(i) for i in range(90)],available

def public_signature_from_query(d):
 return public_signature_from_bank(d,project_street_query(d))
