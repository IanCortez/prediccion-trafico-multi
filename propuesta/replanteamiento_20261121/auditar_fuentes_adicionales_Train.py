"""Availability audit of extra early Train GPS; no feature fitting or outcome selection."""
from base_replanteo import *
from collections import Counter
import csv
access=deny_test();start=time.time();tr=dict(np.load(DATA/'train_x.npz'));known=set(map(int,tr['row']))
cutoff=int(read(OUT/'piloto_transferencia_intervalos_Train_s0/verificacion_publica.json')['cutoff'])
manifest=read(DATA/'manifiesto.json');lat0,lat1,lon0,lon1=manifest['bbox']
source=ROOT/'tesis/porto/porto/train.csv';counts=Counter();duration_hist=Counter();inside_total=0;point_total=0
rng=np.random.default_rng(20261124);valid_aux=0;valid_parent=0;sample_time_hist=Counter();skipped_long_binary=0;max_arrival=0
csv.field_size_limit(2**31-1)
with source.open('rb') as f:
 next(f)
 for row,line in enumerate(f):
  if row in known:continue
  fields=line.split(b',',8);assert len(fields)==9
  stamp=int(fields[5].strip(b'"'))
  if stamp>=cutoff-86400 or fields[7].strip(b'"')==b'True':continue
  # Sampling metadata gives Train membership before any UTF8/CSV/JSON decode.
  binary_n=fields[8].count(b'[')-1
  if binary_n<5:counts['metadata_too_short']+=1;continue
  if binary_n>5761:skipped_long_binary+=1;continue
  arrival=stamp+15*(binary_n-1)
  assert arrival<cutoff
  values=next(csv.reader([line.decode('utf-8')]))
  points=np.asarray(json.loads(values[8]),np.float64);n=len(points)
  assert n==binary_n and int(values[5])==stamp and values[7].lower()=='false'
  assert points.shape==(n,2) and np.isfinite(points).all()
  counts['decoded_additional_earlyTrain_parents']+=1;point_total+=n;max_arrival=max(max_arrival,arrival)
  inside=(points[:,0]>=lon0)&(points[:,0]<=lon1)&(points[:,1]>=lat0)&(points[:,1]<=lat1)
  inside_total+=int(inside.sum())
  if inside.sum()>=5:counts['parents_with_atleast5_core_points']+=1
  duration_hist['<=1785s' if 15*(n-1)<=1785 else '>1785s']+=1
  if not(inside[0] and inside[-1]):counts['atleast_one_fullendpoint_outside_bbox']+=1
  # Fixed prospective source availability probe, not fitted output or cohort filtering.
  found=False;used=set()
  for attempt in range(3):
   if n<6:continue
   maxspan=min(n-2,119);span=int(rng.integers(4,maxspan+1));i=int(rng.integers(0,n-span));j=i+span
   if (i,j) in used:continue
   used.add((i,j))
   if not(inside[i] and inside[j]):continue
   a,b=points[[i,j]];h=np.sin(np.deg2rad(b[1]-a[1])/2)**2+np.cos(np.deg2rad(a[1]))*np.cos(np.deg2rad(b[1]))*np.sin(np.deg2rad(b[0]-a[0])/2)**2
   dist=6371000*2*np.arcsin(min(1,np.sqrt(h)));t=15*span
   if dist<200 or dist/t*3.6>120:continue
   valid_aux+=1;found=True;sample_time_hist[str((t//300)*300)]+=1
  valid_parent+=int(found)
  if counts['decoded_additional_earlyTrain_parents']%50000==0:
   print(json.dumps(dict(stage='extra_Train_availability',parents=counts['decoded_additional_earlyTrain_parents'],valid_interval_samples=valid_aux,seconds=time.time()-start)),flush=True)
result=dict(complete=True,scope='Additional earlyTrain source availability only; no predictor fitted and no primarymetric.',
 source='Original rawCSV, excluding every originalprimaryTrainrow before parse; timestamp+binarypointcount determine safeTrainarrival before decoding.',
 cutoff=cutoff,maximum_selected_source_arrival=max_arrival,all_selected_arrivals_before_heldTrain=True,
 counts=counts,full_duration_histogram=duration_hist,additional_points=point_total,points_inside_primary_bbox=inside_total,
 fixed3attempt_valid_interval_samples=valid_aux,parents_contributing_sample=valid_parent,interval_time_bin_counts=sample_time_hist,
 skipped_long_binary=skipped_long_binary,source_GPS_bytes_persisted=0,
 primary_original_queries_filters_labels_split_unchanged=True,Validation_or_Test_GPS_decoded=False,
 potential_use='Only evidence of extra supervision availability. New training use requires a separate frozen protocol and controls; no accuracy claim.',
 source_sha256_previously_verified=manifest['source_sha256'],code_sha256=sha(__file__),actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(OUT/'auditoria_fuentes_adicionales_Train.json',result)
print(json.dumps(read(OUT/'auditoria_fuentes_adicionales_Train.json')),flush=True)
