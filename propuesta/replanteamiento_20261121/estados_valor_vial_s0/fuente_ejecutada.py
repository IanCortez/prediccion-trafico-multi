"""Fixed Train-only road progress samples. Aux targets never become OD query inputs."""
from base_replanteo import *
import shutil
from consulta_mapa2014 import resources
from numba import set_num_threads
access=deny_test();start=time.time();set_num_threads(2);run=OUT/'estados_valor_vial_s0';assert not run.exists();run.mkdir();shutil.copy2(__file__,run/'fuente_ejecutada.py')
corpus=OUT/'datos_calles_parciales_Train_s0';meta=read(corpus/'resultado.json');assert meta['complete']
raw,c,sample,tree,routes,profiles=resources();set_num_threads(2);D,conn,routing=routes['length']
graph=dict(np.load(OUT/'banco_valor_vial_s0/grafo_estatico.npz'));neighbors=graph['neighbors'];n=1023529;K=4
arrays=dict(chain=np.full((n,K),53766,np.uint16),destination=np.full((n,K),53766,np.uint16),
 fraction=np.zeros((n,K),np.float16),destination_fraction=np.zeros((n,K),np.float16),remaining_km=np.zeros((n,K),np.float16),
 remaining_time_s=np.zeros((n,K),np.float32),local_time_s=np.zeros((n,K),np.float32),local_distance_m=np.zeros((n,K),np.float16),
 route_target=np.full((n,K),-1,np.int8),valid=np.zeros((n,K),bool))
rng=np.random.default_rng(20261122);trips=0;masked_endpoint=0
for item in meta['shards']:
 path=corpus/item['file'];assert sha(path)==item['sha256'];z=dict(np.load(path))
 for i,(a,b) in enumerate(zip(z['offsets'][:-1],z['offsets'][1:])):
  index=int(z['train_index'][i]);eligible=np.flatnonzero(z['remaining_value_valid'][a:b])+a
  if not len(eligible) or z['chain'][b-1]!=z['terminal_chain'][i]:masked_endpoint+=1;continue
  ix=rng.choice(eligible,K,replace=True)
  chain=z['chain'][ix].astype(np.int32);destination=np.full(K,int(z['terminal_chain'][i]),np.int32)
  entry=np.where(ix==a,z['initial_offset_m'][i],0.).astype(np.float64)
  fraction=(entry/c['chain_length'][chain]).astype(np.float16);dfrac=np.float16(z['terminal_offset_m'][i]/c['chain_length'][destination[0]])
  entry=fraction.astype(np.float32)*c['chain_length'][chain];end=float(dfrac)*c['chain_length'][destination]
  remain=c['chain_length'][chain]-entry+D[c['chain_dst'][chain],c['chain_src'][destination]]+end
  same=(chain==destination)&(entry<=end);remain=np.where(same,np.minimum(remain,end-entry),remain)
  good=np.isfinite(remain)&(remain>=-.01)
  if not good.any():continue
  remaining=np.cumsum(z['duration_s'][a:b][::-1],dtype=np.float64)[::-1][ix-a]
  route_target=np.full(K,-1,np.int8)
  for k,j in enumerate(ix):
   if j==b-1:route_target[k]=neighbors.shape[1]
   elif z['next_route_valid'][j]:
    nxt=int(z['chain'][j+1]);where=np.flatnonzero(neighbors[chain[k]]==nxt)
    if len(where):route_target[k]=int(where[0])
  arrays['chain'][index]=chain;arrays['destination'][index]=destination;arrays['fraction'][index]=fraction;arrays['destination_fraction'][index]=dfrac
  arrays['remaining_km'][index]=np.where(good,np.maximum(remain,0)/1000,0).astype(np.float16)
  arrays['remaining_time_s'][index]=remaining.astype(np.float32);arrays['local_time_s'][index]=z['duration_s'][ix]
  arrays['local_distance_m'][index]=z['distance_m'][ix].astype(np.float16);arrays['route_target'][index]=route_target;arrays['valid'][index]=good;trips+=1
 if (len(z['row'])+int(item['file'][6:10])*4096)%65536==0:print(json.dumps(dict(stage='road_progress_targets',trips=trips,seconds=time.time()-start)),flush=True)
assert np.isfinite(arrays['local_distance_m']).all()
file=run/'estados_Train.npz';np.savez_compressed(file,**arrays)
result=dict(complete=True,scope='Four fixedrandom trustedTrainprogressstates per eligible auxiliary trip; no queryfeatures contain actualroad/time/length.',
 primary_n=n,auxiliary_trip_n=trips,valid_state_n=int(arrays['valid'].sum()),masked_endpoint_or_empty_n=masked_endpoint,
 K=K,seed=20261122,fields=list(arrays),time_targets_dtype='Originalsum-preservingfloat32; notfloat16',public_quantities_dtype='Canonicalfloat16fraction/distances',max_outdegree=int(neighbors.shape[1]),
 source_sha256=sha(run/'fuente_ejecutada.py'),corpus_manifest_sha256=sha(corpus/'resultado.json'),targets_sha256=sha(file),
 artifact_MiB=file.stat().st_size/1024**2,Validation_access=False,Test_access=False,seconds=time.time()-start)
assert result['artifact_MiB']<150;write(run/'resultado.json',result);print(json.dumps(result),flush=True)
