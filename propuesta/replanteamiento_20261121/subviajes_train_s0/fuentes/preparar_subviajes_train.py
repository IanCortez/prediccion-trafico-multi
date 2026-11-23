"""Bounded exact-duration subtrip supervision, exclusively eligible Train GPS."""
from base_replanteo import *
import csv,shutil
from collections import Counter
from consulta_subviajes_train import build_training_features
access=deny_test();start=time.time();run=OUT/'subviajes_train_s0'
assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
sources=['preparar_subviajes_train.py','consulta_subviajes_train.py','consulta_historicos_cruzados.py','base_replanteo.py']
for s in sources:shutil.copy2(OUT/s,run/'fuentes'/s)
proof=read(OUT/'verificacion_consulta_subviajes.json')
assert proof['passed'] and proof['source_sha256']==sha(OUT/'consulta_subviajes_train.py')
manifest=read(DATA/'manifiesto.json');tr=dict(np.load(DATA/'train_x.npz'));y=np.load(DATA/'train_y.npy')
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index']
assert len(eligible)==1023485 and len(np.unique(eligible))==len(eligible)
assert np.all(tr['timestamp'][eligible]+y[eligible]<manifest['validation']['start_timestamp'])
rng=np.random.default_rng(20261122);selected=rng.choice(eligible,250000,replace=False)
lookup=np.full(manifest['counts']['raw'],-1,np.int32);lookup[tr['row'][selected]]=selected
source=ROOT/'tesis/porto/porto/train.csv'
assert sha(source)==manifest['source_sha256']
protocol=dict(seed=20261122,max_selected_parents=250000,attempts_per_parent=2,max_aux_examples=500000,
    primary_cohort_labels_and_features_unchanged=True,only_Train_GPS_decoded=True,
    label='15*(j-i) seconds, original GPS sampling interval',
    query=['o_lon','o_lat','d_lon','d_lat','timestamp'],
    auxiliary_only_filter='Original endpoint bbox,OD>=200m,OD speed<=120km/h,time60..1785; proper subsegment; skip UTC-week-crossing parent so parent GPS is wholly excluded by historical week bank.',
    primary_filters_changed=False,whole_week_history_exclusion=True,Validation_and_Test_GPS_decoded=False,
    cache_max_bytes=600*1024**2,source_sha256=manifest['source_sha256'])
write(run/'protocolo.json',protocol)
records=[];targets=[];parent=[];parent_row=[];first=[];last=[];counts=Counter();seen=0
csv.field_size_limit(2**31-1)
with source.open(encoding='utf-8-sig',newline='') as f:
 for row,r in enumerate(csv.DictReader(f)):
  idx=lookup[row]
  if idx<0:continue
  p=np.asarray(json.loads(r['POLYLINE']),np.float64);seen+=1
  ts=int(r['TIMESTAMP']);assert ts==tr['timestamp'][idx] and 15*(len(p)-1)==y[idx]
  assert r['MISSING_DATA'].lower()=='false' and ts+y[idx]<manifest['validation']['start_timestamp']
  if len(p)<6:counts['too_short_parent']+=1;continue
  if ts//604800!=(ts+int(y[idx]))//604800:counts['parent_crosses_UTC_week']+=1;continue
  assert np.isfinite(p).all()
  used=set()
  for _ in range(2):
   span=int(rng.integers(4,len(p)-1));i=int(rng.integers(0,len(p)-span));j=i+span
   assert 4<=j-i<=119 and not (i==0 and j==len(p)-1)
   if (i,j) in used:counts['duplicate_within_parent']+=1;continue
   used.add((i,j));o,d=p[i],p[j];t=15*span
   lat0,lat1,lon0,lon1=manifest['bbox']
   if not (lat0<=o[1]<=lat1 and lat0<=d[1]<=lat1 and lon0<=o[0]<=lon1 and lon0<=d[0]<=lon1):
    counts['endpoint_bbox']+=1;continue
   la,lb=np.deg2rad([o[1],d[1]]);a=np.sin((lb-la)/2)**2+np.cos(la)*np.cos(lb)*np.sin(np.deg2rad(d[0]-o[0])/2)**2
   od=6371000*2*np.arcsin(min(1,np.sqrt(a)))
   if od<200:counts['OD_under200m']+=1;continue
   if od/t*3.6>120:counts['OD_speed_over120']+=1;continue
   query_ts=ts+15*i
   assert query_ts+t==ts+15*j and ts//604800==query_ts//604800
   records.append((o[0],o[1],d[0],d[1],query_ts));targets.append(t)
   parent.append(int(idx));parent_row.append(row);first.append(i);last.append(j)
  if seen%50000==0:print(json.dumps(dict(stage='decode_selected_Train_GPS',parents=seen,aux_examples=len(targets),seconds=time.time()-start)),flush=True)
assert seen==len(selected) and 0<len(targets)<=500000
records=np.asarray(records,np.float64);targets=np.asarray(targets,np.int16)
provenance=dict(parent_train_index=np.asarray(parent,np.int32),parent_row=np.asarray(parent_row,np.int32),i=np.asarray(first,np.uint8),j=np.asarray(last,np.uint8))
np.savez_compressed(run/'consultas_etiquetas_procedencia.npz',o_lon=records[:,0],o_lat=records[:,1],d_lon=records[:,2],d_lat=records[:,3],timestamp=records[:,4].astype(np.int64),y=targets,**provenance)
assert np.array_equal(targets,15*(provenance['j'].astype(int)-provenance['i'].astype(int)))
assert np.all(records[:,4]+targets<manifest['validation']['start_timestamp'])
assert np.all((records[:,4].astype(np.int64)//604800)==(tr['timestamp'][provenance['parent_train_index']]//604800))
write(run/'seleccion.json',dict(parents_selected=seen,aux_examples=len(targets),unique_aux_parents=len(np.unique(provenance['parent_train_index'])),excluded_attempts=counts,query_timestamp_min=int(records[:,4].min()),query_timestamp_max=int(records[:,4].max()),seconds=time.time()-start))
print(json.dumps(dict(stage='construct216_Train_only_features',n=len(targets),seconds=time.time()-start)),flush=True)
keys=protocol['query'];items=[{k:int(v) if k=='timestamp' else float(v) for k,v in zip(keys,row)} for row in records]
rec=read(BEST/'resultado.json');X,names=build_training_features(items,rec['config'])
assert names==rec['features'] and X.shape==(len(targets),216) and np.isfinite(X).all()
assert X.nbytes<=protocol['cache_max_bytes']
np.savez_compressed(run/'features_train.npz',X=X,names=np.asarray(names))
assert (run/'features_train.npz').stat().st_size<=protocol['cache_max_bytes']
result=dict(aux_examples=len(targets),primary_Train_n=1023485,primary_Validation_n=127941,
    original_cohort_unchanged=True,Validation_and_Test_paths_decoded=False,only_selected_Train_paths_JSON_decoded=True,
    all_parent_arrivals_before_Validation=True,all_parent_GPS_excluded_from_own_history_bank=True,
    exact_interval_labels=True,no_provenance_in_query_features=True,feature_shape=X.shape,features=names,
    excluded_attempts=counts,source_sha256={s:sha(run/'fuentes'/s) for s in sources},
    outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in [run/'consultas_etiquetas_procedencia.npz',run/'features_train.npz']},
    actual_data_files_accessed=sorted(access),Test_access=False,seconds=time.time()-start)
write(run/'resultado.json',result);print(json.dumps(result),flush=True)
