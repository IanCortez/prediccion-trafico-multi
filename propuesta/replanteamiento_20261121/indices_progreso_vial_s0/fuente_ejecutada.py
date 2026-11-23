"""Compact Train auxiliary index/endpointlabels for actual recurrent roadprogress; no token duplication."""
from base_replanteo import *
import shutil
access=deny_test();start=time.time();run=OUT/'indices_progreso_vial_s0';assert not run.exists();run.mkdir();shutil.copy2(__file__,run/'fuente_ejecutada.py')
corpus=OUT/'datos_calles_parciales_Train_s0';meta=read(corpus/'resultado.json');assert meta['complete'] and meta['trips']==894105
d=dict(np.load(DATA/'train_x.npz'));y=np.load(DATA/'train_y.npy');eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index'];n=len(y)
v=dict(np.load(DATA/'validation_x.npz'));assert len(eligible)==1023485 and len(v['row'])==127941
assert np.max(d['timestamp'][eligible]+y[eligible])<int(v['timestamp'].min())
with np.load(OUT/'banco_valor_vial_s0/train_proyecciones.npz') as p:bank={key:p[key] for key in ['row','timestamp','chain','fraction','error_m','reachable']}
assert np.array_equal(bank['row'],d['row']) and np.array_equal(bank['timestamp'],d['timestamp'])
c=dict(np.load(DATA/'grafo_compacto_2014.npz'));length=c['chain_length']
shard_index=np.full(n,-1,np.int16);local_trip=np.full(n,-1,np.int16);token_n=np.zeros(n,np.uint16);attachment=np.full(n,-1,np.int8)
time_valid_n=np.zeros(n,np.uint16);next_valid_n=np.zeros(n,np.uint16);value_valid_n=np.zeros(n,np.uint16);all_tokens=0;quality_trips=0;max_sum_error=0.
for shard_id,item in enumerate(meta['shards']):
 path=corpus/item['file'];assert sha(path)==item['sha256'];z=dict(np.load(path));ix=z['train_index'].astype(np.int64)
 assert np.all(shard_index[ix]<0) and np.array_equal(z['row'],d['row'][ix]) and np.array_equal(z['timestamp'],d['timestamp'][ix]) and np.array_equal(z['true_duration_s'],y[ix])
 assert np.all(np.isin(ix,eligible)) and np.all(z['timestamp']+z['true_duration_s']<int(v['timestamp'].min()))
 lengths=np.diff(z['offsets']);assert lengths.max()<65536
 shard_index[ix]=shard_id;local_trip[ix]=np.arange(len(ix),dtype=np.int16);token_n[ix]=lengths.astype(np.uint16)
 oc=bank['chain'][ix,0].astype(np.int32);dc=bank['chain'][ix,1].astype(np.int32)
 op=bank['fraction'][ix,0].astype(np.float64)*length[oc];dp=bank['fraction'][ix,1].astype(np.float64)*length[dc]
 goodo=(oc==z['initial_chain'][:,None])&(np.abs(op-z['initial_offset_m'][:,None])<=50)
 goodd=(dc==z['terminal_chain'][:,None])&(np.abs(dp-z['terminal_offset_m'][:,None])<=50)
 scoreo=(op-z['initial_offset_m'][:,None])**2+bank['error_m'][ix,0].astype(np.float64)**2
 scored=(dp-z['terminal_offset_m'][:,None])**2+bank['error_m'][ix,1].astype(np.float64)**2
 valid=goodo[:,:,None]&goodd[:,None,:]&bank['reachable'][ix];score=np.where(valid,scoreo[:,:,None]+scored[:,None,:],np.inf).reshape(-1,16)
 anyvalid=np.isfinite(score).any(1);attachment[ix[anyvalid]]=score[anyvalid].argmin(1).astype(np.int8)
 for i,(a,b) in enumerate(zip(z['offsets'][:-1],z['offsets'][1:])):
  error=abs(float(z['duration_s'][a:b].astype(np.float64).sum())-float(y[ix[i]]));assert error<.001;max_sum_error=max(max_sum_error,error)
  time_valid_n[ix[i]]=int(np.sum(z['chain'][a:b]!=65535));next_valid_n[ix[i]]=int(z['next_route_valid'][a:b].sum());value_valid_n[ix[i]]=int(z['remaining_value_valid'][a:b].sum())
 all_tokens+=int(z['offsets'][-1]);quality_trips+=len(ix)
file=run/'indices_Train.npz';np.savez_compressed(file,eligible_primary_train_index=eligible,shard_index=shard_index,local_trip=local_trip,token_n=token_n,
 attachment_pair=attachment,time_valid_n=time_valid_n,next_valid_n=next_valid_n,remaining_valid_n=value_valid_n)
assert quality_trips==894105 and all_tokens==41775036 and np.count_nonzero(shard_index>=0)==894105
# Header-onlypublicdistance shape/dtype; neverpersistthismatrix with modelweights.
D=np.load(DATA/'distancias_2014/D.npy',mmap_mode='r')
result=dict(complete=True,scope='FullprimaryTrain index and PRIVATEroadteacher targets only, not a model result.',
 primary_Train=len(eligible),primary_Validation=len(v['row']),auxiliary_trips=quality_trips,tokens=all_tokens,
 attachment_supervised_n=int((attachment>=0).sum()),attachment_teacher_coverage=float((attachment[shard_index>=0]>=0).mean()),
 token_length_quantiles=np.quantile(token_n[shard_index>=0],[.5,.9,.95,.99,.999,1]),max_auxiliary_sum_error_s=max_sum_error,
 time_valid_tokens=int(time_valid_n.astype(np.int64).sum()),next_valid_tokens=int(next_valid_n.astype(np.int64).sum()),
 remaining_valid_tokens=int(value_valid_n.astype(np.int64).sum()),primary_queries_filtered_out=0,
 public_distance_matrix=dict(shape=list(D.shape),dtype=str(D.dtype),bytes=int(D.nbytes),checkpoint_persistent=False),
 no_duplicate_full_tokenbank=True,corpus_manifest_sha256=sha(corpus/'resultado.json'),source_sha256=sha(run/'fuente_ejecutada.py'),
 outputs_sha256=sha(file),artifact_MiB=file.stat().st_size/1024**2,no_Validationlabels_or_paths_or_Test_accessed=True,seconds=time.time()-start)
write(run/'resultado.json',result);print(json.dumps({k:result[k] for k in ['complete','primary_Train','auxiliary_trips','attachment_teacher_coverage','token_length_quantiles','artifact_MiB','seconds']}),flush=True)
