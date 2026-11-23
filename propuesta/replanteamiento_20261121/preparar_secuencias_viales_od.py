"""Bounded Train/Validation public-map token cache; no observed query routes."""
from base_replanteo import *
import shutil
from inferencia import feature_input
from secuencias_viales_od import build_public_sequences
access=deny_test();start=time.time();run=OUT/'secuencias_viales_od_s0'
assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
for name in ['preparar_secuencias_viales_od.py','secuencias_viales_od.py','consulta_historicos_cruzados.py','base_replanteo.py']:
 shutil.copy2(OUT/name,run/'fuentes'/name)
write(run/'protocolo.json',dict(query=['o_lon','o_lat','d_lon','d_lat','timestamp'],
 routes='Two public2014routes: length-optimal and freeflow-optimal. Same projection and selection as protected216features.',
 tokens_per_route=32,sampling='Edge at each uniform arc-length midpoint; no observed path or observed point count.',
 missing_route='All zero tokens plus existing protected global-feature fallback; no rows removed.',
 new_information='Fine ordered road composition beyond seven geometric samples and route-level aggregates.',
 planned_model='Original protected global mixture plus ordered road attention branch, trained on unchanged original primary durations.',
 max_cache_bytes=500*1024**2,Train_only_confidence=True,Validation_labels_accessed=False,Test='sealed'))
rec=read(BEST/'resultado.json');keys=['o_lon','o_lat','d_lon','d_lat','timestamp']
# Certify a small Train sample before producing any large cache.
tr,X,names=feature_input('train',rec);rng=np.random.default_rng(20261122);ii=np.sort(rng.choice(len(X),65,replace=False))
def items(d,indices):
 return [{k:int(d[k][i]) if k=='timestamp' else float(d[k][i]) for k in keys} for i in indices]
q=items(tr,ii);tokens,s=build_public_sequences(q);rev,rs=build_public_sequences(q[::-1])
assert np.array_equal(tokens,rev[::-1]) and np.array_equal(s,rs[::-1])
single,ss=build_public_sequences([q[0]]);assert np.array_equal(single,tokens[:1]) and np.array_equal(ss,s[:1])
max_error=0.
for j,kind in enumerate(['length','free']):
 valid=s[:,j,2]>0
 for col,name,scale in [(0,'projected_'+kind+'_polyline_km',1000),(1,'projected_'+kind+'_freeflow_600s',600)]:
  error=float(np.max(np.abs(s[valid,j,col]-X[ii[valid],names.index(name)]))*scale) if valid.any() else 0.
  assert error<.1,(kind,name,error);max_error=max(max_error,error)
 rejected=False
 try:build_public_sequences([dict(q[0],actual_path=[])])
 except AssertionError:rejected=True
 assert rejected
write(run/'verificacion_entrada.json',dict(passed=True,Train65_route_summaries_match_protected_features=True,max_error_m_or_s=max_error,
 reversal_and_singleton_exact=True,actual_path_extra_field_rejected=True,Validation_labels_accessed=False,Test_access=False))
print(json.dumps(dict(stage='public_route_CPU_gate',passed=True,max_error=max_error)),flush=True)
del X
outputs={};cache_bytes=0
for split in ['train','validation']:
 d=dict(np.load(DATA/(split+'_x.npz')));n=len(d['row'])
 # Chunk query projection to bound temporary Python dictionaries; no Dijkstra reruns.
 T=np.empty((n,2,32),np.int32);S=np.empty((n,2,4),np.float32)
 for first in range(0,n,100000):
  last=min(n,first+100000);T[first:last],S[first:last]=build_public_sequences(items(d,range(first,last)))
  print(json.dumps(dict(stage='public_route_tokens',split=split,done=last,n=n,seconds=time.time()-start)),flush=True)
 assert T.shape==(n,2,32) and np.isfinite(S).all()
 path=run/(split+'_secuencias.npz');np.savez_compressed(path,row=d['row'],timestamp=d['timestamp'],tokens=T,summary=S)
 cache_bytes+=path.stat().st_size;assert cache_bytes<=500*1024**2
 outputs[path.name]=dict(bytes=path.stat().st_size,sha256=sha(path),n=n,missing_routes_per_kind=(S[:,:,2]==0).sum(0))
 if split=='train':
  eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index']
  counts=np.bincount(T[eligible].ravel(),minlength=174113);counts[0]=0
  np.save(run/'frecuencias_Train.npy',counts.astype(np.uint32))
 del T,S
result=dict(outputs=outputs,cache_bytes=cache_bytes,source_sha256={p.name:sha(p) for p in (run/'fuentes').glob('*.py')},
 Validation_labels_accessed=False,Test_access=False,observed_Validation_or_Test_paths_decoded=False,
 all_primary_rows_retained=True,query_fields=keys,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
assert not any(Path(p).name=='validation_y.npy' for p in access)
write(run/'resultado.json',result);print(json.dumps(dict(stage='complete',cache_bytes=cache_bytes,seconds=result['seconds'],Test_access=False)),flush=True)
