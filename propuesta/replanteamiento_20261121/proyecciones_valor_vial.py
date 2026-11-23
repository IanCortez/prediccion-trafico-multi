"""Standalone public raw5 street projections; same canonical arithmetic as prepared bank."""
from base_replanteo import *
from numba import set_num_threads
from consulta_mapa2014 import resources,ORIGIN
from map_matching import project_candidates
def project_street_query(d):
 raw,c,sample,tree,routes,profiles=resources();set_num_threads(2);D,conn,routing=routes['length']
 flat=c['flat_incoming_arc'];valid=flat>=0;pc=np.repeat(np.arange(len(c['chain_src']),dtype=np.int32),np.diff(c['chain_offsets']))
 edge_chain=np.full(len(raw['u']),-1,np.int32);edge_chain[flat[valid]]=pc[valid]
 edge_position=np.empty(len(raw['u']),np.int32);edge_position[flat[valid]]=np.flatnonzero(valid);prefix=c['prefix_length'][edge_position-1]
 ll=np.stack([np.column_stack([d['o_lon'],d['o_lat']]),np.column_stack([d['d_lon'],d['d_lat']])],1).reshape(-1,2)
 points=np.column_stack([(ll[:,0]-ORIGIN[0])*111195*np.cos(np.deg2rad(ORIGIN[1])),(ll[:,1]-ORIGIN[1])*111195])
 _,near=tree.query(points,k=32,workers=2);edge,frac,error=project_candidates(points,near,sample,raw['xy'],raw['u'],raw['v'])
 chain=edge_chain[edge].reshape(-1,2,4);pos=(prefix[edge]+frac*raw['length'][edge]).reshape(-1,2,4);length=c['chain_length'][chain]
 fraction=np.divide(pos,length,out=np.zeros_like(pos),where=length>0).astype(np.float16);pos=fraction.astype(np.float32)*length
 oc,dc=chain[:,0],chain[:,1];op,dp=pos[:,0],pos[:,1]
 remain=length[:,0,:,None]-op[:,:,None]+D[c['chain_dst'][oc][:,:,None],c['chain_src'][dc][:,None,:]]+dp[:,None,:]
 same=(oc[:,:,None]==dc[:,None,:])&(op[:,:,None]<=dp[:,None,:]);remain=np.where(same,np.minimum(remain,dp[:,None,:]-op[:,:,None]),remain)
 good=np.isfinite(remain)&(remain>=-.01);remain=np.where(good,np.maximum(remain,0)/1000,0)
 return dict(chain=chain.astype(np.uint16),fraction=fraction,error_m=error.reshape(-1,2,4).astype(np.float16),remaining_km=remain.astype(np.float16),reachable=good)
