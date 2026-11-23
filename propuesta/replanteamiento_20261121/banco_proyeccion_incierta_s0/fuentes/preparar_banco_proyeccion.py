"""Canonical FP16 route-hypothesis bank, unchanged Train/Validation query cohorts."""
from base_replanteo import *
import shutil
from rutas_proyeccion_incierta import build_uncertain_routes
access=deny_test();start=time.time();run=OUT/'banco_proyeccion_incierta_s0'
assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
proof=read(OUT/'piloto_proyeccion_incierta_s0_v2/resultado.json')
assert proof['passed'] and proof['source_sha256']['rutas_proyeccion_incierta.py']==sha(OUT/'rutas_proyeccion_incierta.py')
for name in ['preparar_banco_proyeccion.py','rutas_proyeccion_incierta.py','consulta_historicos_cruzados.py','base_replanteo.py']:
 shutil.copy2(OUT/name,run/'fuentes'/name)
protocol=dict(query=['o_lon','o_lat','d_lon','d_lat','timestamp'],routes=8,features_per_route=31,
 precision='Canonical FP16 storage/features cast toFP32 in model; public raw-query adapter uses same quantization.',
 candidate_selection='Protected default length/free routes, plus two cost types for three other distinct projection pairs selected by squared GPS residual. No label selection.',
 static_features='14relative shape coordinates,12route summaries; column24 planning regret, column25availability. Last5historical movement summaries.',
 history='Train own UTC departure-week bank excluded at all levels; full eligibleTrain history for heldout queries.',
 max_cache_bytes=500*1024**2,primary_filters_and_labels_unchanged=True,Validation_labels_accessed=False,Test='sealed')
write(run/'protocolo.json',protocol)
keys=protocol['query'];outputs={};cache_bytes=0;norm_sum=np.zeros(31,np.float64);norm_square=np.zeros(31,np.float64);norm_count=0;quant_error=0.
for split in ['train','validation']:
 d=dict(np.load(DATA/(split+'_x.npz')));n=len(d['row']);bank=np.empty((n,8,31),np.float16)
 pairs=np.empty((n,8,2),np.int8);available_count=np.zeros(8,np.int64)
 if split=='train':
  eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index'];emask=np.zeros(n,bool);emask[eligible]=True;assert emask.sum()==1023485
 for first in range(0,n,50000):
  last=min(first+50000,n);items=[{k:int(d[k][i]) if k=='timestamp' else float(d[k][i]) for k in keys} for i in range(first,last)]
  B,P=build_uncertain_routes(items,training=split=='train');H=B.astype(np.float16);assert np.isfinite(H).all()
  quant_error=max(quant_error,float(np.max(np.abs(H.astype(np.float32)-B))))
  bank[first:last]=H;pairs[first:last]=P;available_count+=(H[:,:,20]>0).sum(0)
  if split=='train':
   chunk=H[emask[first:last]].astype(np.float64);good=chunk[:,:,20]>0;values=chunk[good]
   norm_sum+=values.sum(0);norm_square+=np.square(values).sum(0);norm_count+=len(values)
  print(json.dumps(dict(stage='projection_bank',split=split,done=last,n=n,seconds=time.time()-start)),flush=True)
 if split=='train':
  mean=norm_sum/norm_count;std=np.sqrt(np.maximum(norm_square/norm_count-mean**2,0)).clip(1e-5)
  np.savez_compressed(run/'normalizacion_Train.npz',mean=mean.astype(np.float32),std=std.astype(np.float32),available_routes=norm_count)
 path=run/(split+'_banco.npz');np.savez_compressed(path,row=d['row'],timestamp=d['timestamp'],bank=bank,projection_pairs=pairs)
 cache_bytes+=path.stat().st_size;assert cache_bytes<=protocol['max_cache_bytes']
 outputs[path.name]=dict(bytes=path.stat().st_size,sha256=sha(path),n=n,available_per_slot=available_count)
 del bank,pairs
result=dict(outputs=outputs,cache_bytes=cache_bytes,canonical_precision='FP16features,FP32model',max_absolute_feature_quantization_error=quant_error,
 normalization='Mean/std of available routes in1023485eligibleTrain rows, fitted before validation; no target input.',
 normalization_sha256=sha(run/'normalizacion_Train.npz'),all_primary_rows_retained=True,original_query_fields_only=True,
 Validation_labels_accessed=False,Test_access=False,observed_heldout_paths_decoded=False,
 source_sha256={p.name:sha(p) for p in (run/'fuentes').glob('*.py')},actual_data_files_accessed=sorted(access),seconds=time.time()-start)
assert not any(Path(p).name=='validation_y.npy' for p in access)
write(run/'resultado.json',result)
print(json.dumps(dict(stage='complete',cache_MiB=cache_bytes/1024**2,seconds=result['seconds'],Test_access=False)),flush=True)
