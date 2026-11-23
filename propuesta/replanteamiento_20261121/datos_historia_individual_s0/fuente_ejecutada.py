"""Compact64source index cache, derived only from five-field queries and eligibleTrain memory."""
from base_replanteo import *
import shutil,pandas as pd
from scipy.spatial import cKDTree
from historicos_rutas import query_xy
access=deny_test();start=time.time();run=OUT/'datos_historia_individual_s0';assert not run.exists();run.mkdir()
shutil.copy2(__file__,run/'fuente_ejecutada.py')
gate=read(OUT/'piloto_compresion_historial_s0/resultado.json');assert gate['passed']
memory=OUT/'datos_prototipos_historicos_s0/memoria_Train.npz';meta=read(OUT/'datos_prototipos_historicos_s0/resultado.json')
assert sha(memory)==meta['memory']['sha256'];m=dict(np.load(memory));trees={}
protocol=dict(query=['o_lon','o_lat','d_lon','d_lat','timestamp'],source_UTCweekfold_excluded_for_Train=True,
 Val_memory='Only eligibleTrain trips arriving before firstValidation departure.',
 neighbors=64,cutoff_OD4D_km=.5,weight_clock_sigma_hours=4,weight_distance_sigma_km=.25,weekend_mismatch_weight=.35,
 cache='Only source indices, five-field query derivatives,row/timestamp. No query labels or query trajectory.',
 max_total_MiB_including_shared_memory=450,normalization='Pending Train-only raw-token feature constructor; no neural model run by this preparer.',
 Test='sealed')
write(run/'protocolo.json',protocol);outputs={}
for split in ['train','validation']:
 d=dict(np.load(DATA/(split+'_x.npz')));n=len(d['timestamp']);xy=query_xy(d)
 local=pd.to_datetime(d['timestamp'],unit='s',utc=True).tz_convert('Europe/Lisbon')
 hour=(np.asarray(local.hour)+np.asarray(local.minute)/60+np.asarray(local.second)/3600).astype(np.float32);weekend=np.asarray(local.dayofweek)>=5
 folds=(d['timestamp']//604800)%5 if split=='train' else np.full(n,5);indices=np.full((n,64),-1,np.int32)
 if split=='validation':assert np.max(m['timestamp']+m['time_s'])<int(d['timestamp'].min())
 for f in np.unique(folds):
  f=int(f);take=np.flatnonzero(folds==f)
  if f not in trees:
   source=np.flatnonzero(m['fold']!=f) if f<5 else np.arange(len(m['row']))
   trees[f]=(cKDTree(m['xy'][source]),source)
  tree,source=trees[f];distance,near=tree.query(xy[take],k=64,workers=2);idx=source[near].astype(np.int32);idx[distance>.5]=-1
  valid=idx>=0
  if f<5:assert (m['fold'][idx[valid]]!=f).all()
  indices[take]=idx
  print(json.dumps(dict(stage='index_memory_only',split=split,fold=f,rows=len(take),seconds=time.time()-start)),flush=True)
 file=run/(split+'_indices.npz')
 np.savez_compressed(file,row=d['row'],timestamp=d['timestamp'],xy_km=xy.astype(np.float32),hour=hour,weekend=weekend,indices=indices)
 outputs[file.name]=dict(n=n,shape=indices.shape,sha256=sha(file),bytes=file.stat().st_size,
  fraction_with_history=float(np.mean((indices>=0).any(1))),query_labels_used=False)
total=sum(q.stat().st_size for q in run.iterdir() if q.is_file())+memory.stat().st_size
assert total<450*1024**2
result=dict(outputs=outputs,source_sha256=sha(run/'fuente_ejecutada.py'),memory_sha256=sha(memory),shared_memory_file=str(memory),
 total_bytes_including_shared_memory=total,gate_sha256=sha(OUT/'piloto_compresion_historial_s0/resultado.json'),
 all_Train_sources_exclude_query_UTCweekfold=True,all_Validation_sources_arrive_before_Validation=True,
 query_labels_or_geometry_used=False,Validation_labels_accessed=False,Test_access=False,
 ready_for_neural_training=False,next_required='Define/verify raw28token constructor, fit Train-only normalization, then train frozen-parent attention mixture and conditional compression control.',
 actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(run/'resultado.json',result);print(json.dumps(dict(stage='complete',MiB_including_shared_memory=total/1024**2,seconds=result['seconds'])),flush=True)
