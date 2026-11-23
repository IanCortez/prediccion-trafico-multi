"""Public raw5 street projection bank, static graph and Train-only normalization."""
from base_replanteo import *
import shutil
from numba import set_num_threads
from consulta_mapa2014 import resources,ORIGIN
from map_matching import project_candidates
from inferencia import feature_input
access=deny_test();start=time.time();set_num_threads(2);folder=OUT/'banco_valor_vial_s0';assert not folder.exists();folder.mkdir();shutil.copy2(__file__,folder/'fuente_ejecutada.py')
assert read(OUT/'oraculo_tiempo_calle_train_s0/resultado.json')['passed']
raw,c,sample,tree,routes,profiles=resources();set_num_threads(2);D,conn,routing=routes['length']
flat=c['flat_incoming_arc'];valid=flat>=0;pc=np.repeat(np.arange(len(c['chain_src']),dtype=np.int32),np.diff(c['chain_offsets']))
edge_chain=np.full(len(raw['u']),-1,np.int32);edge_chain[flat[valid]]=pc[valid]
edge_position=np.empty(len(raw['u']),np.int32);edge_position[flat[valid]]=np.flatnonzero(valid);edge_prefix=c['prefix_length'][edge_position-1]
count=len(c['chain_src']);assert count==53766
outgoing=[np.flatnonzero(c['chain_src']==node).astype(np.int32) for node in range(len(c['junction_raw']))]
max_degree=max(map(len,outgoing));neighbors=np.full((count,max_degree),count,np.int32);neighbor_valid=np.zeros((count,max_degree),bool)
for chain in range(count):
 nexts=outgoing[c['chain_dst'][chain]];neighbors[chain,:len(nexts)]=nexts;neighbor_valid[chain,:len(nexts)]=True
srcxy=raw['xy'][c['junction_raw'][c['chain_src']]]/1000;dstxy=raw['xy'][c['junction_raw'][c['chain_dst']]]/1000;v=dstxy-srcxy;norm=np.linalg.norm(v,axis=1).clip(.001)
speed=c['chain_length']/np.maximum(c['chain_free'],.001);degree=np.asarray([len(outgoing[n]) for n in c['chain_dst']])
static=np.column_stack([srcxy,dstxy,(srcxy+dstxy)/2,v/norm[:,None],np.log1p(c['chain_length'])/10,np.log1p(c['chain_free'])/6,speed/20,degree/10])
assert static.shape==(count,12) and np.isfinite(static).all()
np.savez_compressed(folder/'grafo_estatico.npz',static=static.astype(np.float32),neighbors=neighbors,neighbor_valid=neighbor_valid,
 chain_length=c['chain_length'].astype(np.float32),chain_src=c['chain_src'],chain_dst=c['chain_dst'])
meta=dict(scope='Public2014map projections fromfivequeryfields only; no actualquerypath/N/duration used.',query=['o_lon','o_lat','d_lon','d_lat','timestamp'],outputs={},chain_n=count,max_outdegree=max_degree,Test=False)
for split in ['train','validation']:
 d=dict(np.load(DATA/(split+'_x.npz')));n=len(d['row']);chains=[];fractions=[];errors=[];distances=[];reachable=[]
 for a in range(0,n,16384):
  b=min(a+16384,n);ll=np.stack([np.column_stack([d['o_lon'][a:b],d['o_lat'][a:b]]),np.column_stack([d['d_lon'][a:b],d['d_lat'][a:b]])],1).reshape(-1,2)
  points=np.column_stack([(ll[:,0]-ORIGIN[0])*111195*np.cos(np.deg2rad(ORIGIN[1])),(ll[:,1]-ORIGIN[1])*111195])
  _,near=tree.query(points,k=32,workers=2);edge,frac,error=project_candidates(points,near,sample,raw['xy'],raw['u'],raw['v'])
  chain=edge_chain[edge].reshape(-1,2,4);pos=(edge_prefix[edge]+frac*raw['length'][edge]).reshape(-1,2,4)
  length=c['chain_length'][chain];fraction=np.divide(pos,length,out=np.zeros_like(pos),where=length>0)
  # Canonical float16 quantities are used consistently for bank and raw inference.
  fraction=fraction.astype(np.float16);pos=fraction.astype(np.float32)*length
  oc,dc=chain[:,0],chain[:,1];op,dp=pos[:,0],pos[:,1]
  remain=length[:,0,:,None]-op[:,:,None]+D[c['chain_dst'][oc][:,:,None],c['chain_src'][dc][:,None,:]]+dp[:,None,:]
  same=(oc[:,:,None]==dc[:,None,:])&(op[:,:,None]<=dp[:,None,:])
  remain=np.where(same,np.minimum(remain,dp[:,None,:]-op[:,:,None]),remain)
  good=np.isfinite(remain)&(remain>=-.01)
  remain=np.where(good,np.maximum(remain,0)/1000,0)
  chains.append(chain.astype(np.uint16));fractions.append(fraction);errors.append(error.reshape(-1,2,4).astype(np.float16))
  distances.append(remain.astype(np.float16));reachable.append(good)
 chain=np.concatenate(chains);fraction=np.concatenate(fractions);error=np.concatenate(errors);distance=np.concatenate(distances);good=np.concatenate(reachable)
 file=folder/(split+'_proyecciones.npz');np.savez_compressed(file,row=d['row'],timestamp=d['timestamp'],chain=chain,fraction=fraction,error_m=error,remaining_km=distance,reachable=good)
 meta['outputs'][file.name]=dict(sha256=sha(file),n=n,bytes=file.stat().st_size,no_reachable_pair_n=int((~good.reshape(n,-1).any(1)).sum()))
 print(json.dumps(dict(stage='public_projection_bank',split=split,n=n,seconds=time.time()-start)),flush=True)
rec=read(BEST/'resultado.json');d,X,names=feature_input('train',rec);keep=np.asarray([not n.startswith('mm_') for n in names]);assert keep.sum()==206
ids=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index'];public=X[ids][:,keep];mean=public.mean(0,dtype=np.float64).astype(np.float32);std=public.std(0,dtype=np.float64).clip(.01).astype(np.float32)
np.savez_compressed(folder/'normalizacion_Train.npz',mean=mean,std=std);meta['features']=[n for n,k in zip(names,keep) if k];meta['normalization_sha256']=sha(folder/'normalizacion_Train.npz')
meta.update(static_graph_sha256=sha(folder/'grafo_estatico.npz'),source_sha256=sha(folder/'fuente_ejecutada.py'),artifact_MiB=sum(p.stat().st_size for p in folder.iterdir() if p.is_file())/1024**2,seconds=time.time()-start,
 primary_Train_queries=1023485,primary_Validation_queries=127941,no_Validation_labels_read=True,actual_data_files_accessed=sorted(access))
assert meta['artifact_MiB']<150;write(folder/'resultado.json',meta);print(json.dumps({k:meta[k] for k in ['artifact_MiB','seconds','chain_n','max_outdegree']}),flush=True)
