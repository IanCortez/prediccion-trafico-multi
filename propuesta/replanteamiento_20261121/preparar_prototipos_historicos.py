"""Bounded source-safe geometry/time prototype bank; no heldout labels or paths."""
from base_replanteo import *
import shutil
from historicos_rutas import HistoryRouteBank,make_memory,HISTORY_NAMES
from consulta_od import od_records
access=deny_test();start=time.time();run=OUT/'datos_prototipos_historicos_s0';assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
proof=read(OUT/'piloto_prototipos_historicos_s0/resultado.json');assert proof['passed']
assert proof['source_sha256']==sha(OUT/'piloto_prototipos_ruta.py')
for name in ['preparar_prototipos_historicos.py','historicos_rutas.py','piloto_prototipos_ruta.py','rutas_proyeccion_incierta.py','consulta_historicos_cruzados.py','base_replanteo.py']:
 shutil.copy2(OUT/name,run/'fuentes'/name)
write(run/'protocolo.json',dict(query=['o_lon','o_lat','d_lon','d_lat','timestamp'],prototypes=6,features=HISTORY_NAMES,
 retrieval='64neighbors in4Dendpoints,Gaussian250m/4hour/daytype;500mcutoff affects memory availability only, not primary cohort.',
 selection='Maximumweight first, then5greedy weighted geometric coverage prototypes. Query geometry never supplied.',
 time_prior='Weighted median/MAD of OD-scaled durations from other-fold source trips assigned by historical geometry to6medoids.',
 no_neighbor_cached_movement_profiles=True,own_UTCweek_fold_excluded_from_all_retrieved_labels_and_statistics=True,
 precision='CanonicalFP16history inputs; FP32model, original216andpublic8inputs unchanged.',
 max_memory_plus_cache_bytes=450*1024**2,Validation_labels_accessed=False,Test='sealed'))
memory_path=run/'memoria_Train.npz';mem=make_memory(memory_path);library=HistoryRouteBank(memory_path)
write(run/'memoria_manifiesto.json',mem)
tr=dict(np.load(DATA/'train_x.npz'));public=np.load(OUT/'banco_proyeccion_incierta_s0/train_banco.npz')['bank']
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index'];f=int((tr['timestamp'][eligible[0]]//604800)%5)
ii=eligible[((tr['timestamp'][eligible]//604800)%5)==f][:5];keys=['o_lon','o_lat','d_lon','d_lat','timestamp']
items=[{k:int(tr[k][i]) if k=='timestamp' else float(tr[k][i]) for k in keys} for i in ii]
d=od_records(items);before,rows=library._build(d,public[ii],True)
own=library.memory['fold']==f;saved_time=library.memory['time_s'][own].copy();saved_geom=library.memory['geometry'][own].copy()
library.memory['time_s'][own]+=100000;library.memory['geometry'][own]+=100
after,arows=library._build(d,public[ii],True)
assert np.array_equal(before,after) and np.array_equal(rows,arows)
library.memory['time_s'][own]=saved_time;library.memory['geometry'][own]=saved_geom
raw,rrows=library.build(items,training=True);reverse,_=library.build(items[::-1],training=True);single,_=library.build([items[0]],training=True)
assert np.array_equal(raw,before) and np.array_equal(rows,rrows) and np.array_equal(raw,reverse[::-1]) and np.array_equal(raw[:1],single)
extra_rejected=False
try:library.build([dict(items[0],duration=123)],training=True)
except AssertionError:extra_rejected=True
assert extra_rejected
write(run/'verificacion_entrada.json',dict(passed=True,own_fold_time_and_geometry_intervention_invariant=True,
 raw5_vs_cached_public_defaults_exact=True,reverse_and_singleton_exact=True,query_target_extra_field_rejected=True,
 validation_labels_accessed=False,Test_access=False))
print(json.dumps(dict(stage='historical_input_gate',passed=True,memory_n=mem['n'],seconds=time.time()-start)),flush=True)
outputs={};total_bytes=mem['bytes'];ss=np.zeros(26,np.float64);square=np.zeros(26,np.float64);count=0
train_top_time=None
for split in ['train','validation']:
 d=dict(np.load(DATA/(split+'_x.npz')));n=len(d['row'])
 if split!='train':public=np.load(OUT/'banco_proyeccion_incierta_s0/validation_banco.npz')['bank']
 H=np.empty((n,6,26),np.float16);source_rows=np.empty((n,6),np.int32)
 if split=='train':
  emask=np.zeros(n,bool);emask[eligible]=True
 for first in range(0,n,20000):
  last=min(first+20000,n);raw={k:d[k][first:last] for k in keys}
  # Public defaults are already certified from the same five frozen query fields.
  chunk,sources=library._build(raw,public[first:last],training=split=='train')
  assert np.isfinite(chunk).all();H[first:last]=chunk;source_rows[first:last]=sources
  if split=='train':
   Z=chunk[emask[first:last]].astype(np.float64);available=Z[:,:,23]>.5;values=Z[available];ss+=values.sum(0);square+=np.square(values).sum(0);count+=len(values)
  if first%100000==0 or last==n:print(json.dumps(dict(stage='historical_bank',split=split,done=last,n=n,seconds=time.time()-start)),flush=True)
 if split=='train':
  mean=ss/count;std=np.sqrt(np.maximum(square/count-mean**2,0)).clip(1e-5)
  prior=H[:,:,16].astype(np.float32);mass=H[:,:,18].astype(np.float32);best=mass.argmax(1);tprior=prior[np.arange(n),best]*600
  valid=emask&(mass.max(1)>0)&(tprior>0);y=np.load(DATA/'train_y.npy')
  beta,intercept=np.polyfit(np.log(tprior[valid]),np.log(y[valid]),1)
  intercept+=np.log(np.median(y[valid]/np.exp(intercept+beta*np.log(tprior[valid]))))
  assert np.isfinite(beta) and np.isfinite(intercept) and beta>0
  np.savez_compressed(run/'constructor_Train.npz',mean=mean.astype(np.float32),std=std.astype(np.float32),
   time_anchor=np.array([intercept,beta],np.float32),n=valid.sum(),normalization_routes=count)
 path=run/(split+'_historicos.npz');np.savez_compressed(path,row=d['row'],timestamp=d['timestamp'],history=H,source_rows=source_rows)
 total_bytes+=path.stat().st_size;assert total_bytes<=450*1024**2
 outputs[path.name]=dict(n=n,bytes=path.stat().st_size,sha256=sha(path),fraction_with_history=float(np.mean((H[:,:,23]>.5).any(1))))
 del H,source_rows,public
result=dict(memory=mem,outputs=outputs,total_bytes=total_bytes,features=HISTORY_NAMES,all_primary_rows_retained=True,
 own_fold_never_used_in_retrieved_geometry_or_time=True,own_fold_intervention_passed=True,
 constructor_fitted_only_on_eligible_Train=True,constructor_sha256=sha(run/'constructor_Train.npz'),
 Validation_labels_accessed=False,Test_access=False,observed_heldout_paths_decoded=False,
 sources={p.name:sha(p) for p in (run/'fuentes').glob('*.py')},actual_data_files_accessed=sorted(access),seconds=time.time()-start)
assert not any(Path(p).name=='validation_y.npy' for p in access)
write(run/'resultado.json',result)
print(json.dumps(dict(stage='complete',total_MiB=total_bytes/1024**2,seconds=result['seconds'],Test_access=False)),flush=True)
