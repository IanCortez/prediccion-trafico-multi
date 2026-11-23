"""Bounded raw Train point bank for implicit subtrajectory retrieval, no pair explosion."""
from base_replanteo import *
import csv,shutil
access=deny_test();start=time.time();folder=OUT/'memoria_puntos_Train_s0';assert not folder.exists();folder.mkdir();(folder/'fuentes').mkdir()
shutil.copy2(__file__,folder/'fuentes'/Path(__file__).name)
tr=dict(np.load(DATA/'train_x.npz'));y=np.load(DATA/'train_y.npy');manifest=read(DATA/'manifiesto.json')
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index'];n=len(y)
assert len(eligible)==1023485 and np.all(tr['timestamp'][eligible]+y[eligible]<manifest['validation']['start_timestamp'])
sizes=np.zeros(n,np.int16);sizes[eligible]=(y[eligible]/15+1).astype(np.int16);assert sizes.max()==120
offsets=np.r_[0,np.cumsum(sizes,dtype=np.int64)];point_n=int(offsets[-1])
xy=np.empty((point_n,2),np.float32);seen=np.zeros(n,bool)
lookup=np.full(manifest['counts']['raw'],-1,np.int32);lookup[tr['row'][eligible]]=eligible
source=ROOT/'tesis/porto/porto/train.csv';csv.field_size_limit(2**31-1)
sx=111195*np.cos(np.deg2rad(41.16));sy=111195.
count=0;max_endpoint_error=0.
with source.open('rb') as f:
 header=next(csv.reader([next(f).decode('utf-8-sig').strip()]));pc=header.index('POLYLINE');tc=header.index('TIMESTAMP');mc=header.index('MISSING_DATA')
 for rawrow,line in enumerate(f):
  ti=int(lookup[rawrow])
  if ti<0:continue
  # Trainmembership checked before ANY nonselected row is decoded or parsed.
  fields=next(csv.reader([line.decode('utf-8')]));points=np.asarray(json.loads(fields[pc]),np.float64)
  assert int(fields[tc])==tr['timestamp'][ti] and fields[mc].lower()=='false'
  assert len(points)==sizes[ti] and 15*(len(points)-1)==y[ti]
  assert not seen[ti] and np.isfinite(points).all()
  expected=np.array([[tr['o_lon'][ti],tr['o_lat'][ti]],[tr['d_lon'][ti],tr['d_lat'][ti]]])
  error=float(np.max(np.abs(points[[0,-1]]-expected)));assert error<1e-10;max_endpoint_error=max(max_endpoint_error,error)
  a,b=offsets[ti:ti+2]
  xy[a:b,0]=(points[:,0]+8.62)*sx;xy[a:b,1]=(points[:,1]-41.16)*sy
  seen[ti]=True;count+=1
  if count%100000==0:print(json.dumps(dict(stage='decode_Train_only',parents=count,points_written=int(sizes[seen].sum()),seconds=time.time()-start)),flush=True)
  if count==len(eligible):break
assert count==1023485 and seen[eligible].all() and np.isfinite(xy).all()
lat0,lat1,lon0,lon1=manifest['bbox']
bounds=np.array([(lon0+8.62)*sx,(lat0-41.16)*sy,(lon1+8.62)*sx,(lat1-41.16)*sy])
indexable=(xy[:,0]>=bounds[0])&(xy[:,0]<=bounds[2])&(xy[:,1]>=bounds[1])&(xy[:,1]<=bounds[3])
file=folder/'puntos_Train.npz'
np.savez_compressed(file,xy_m=xy,offsets=offsets,parent_gps_n=sizes,indexable_packed=np.packbits(indexable),
 row=tr['row'],timestamp=tr['timestamp'],time_s=y,eligible=eligible,
 parent_fold=((tr['timestamp']//604800)%5).astype(np.int8),bounds_m=bounds)
assert file.stat().st_size<500*1024**2
result=dict(complete=True,scope='HistoricalTrainGPSpoints only; noqueryobservedtrajectory. Storepoints/parentoffsets, never enumerateallsubtrippairs.',
 original_primary_Train=1023485,primary_queries_filtered_out=0,source_parents=count,source_points=point_n,indexable_source_points=int(indexable.sum()),
 projection='Metersrelative(-8.62,41.16);float32millimeter-scale precision inthisarea.',
 future_source_labels='15*(j-i)seconds fororderedpoints i<j fromSAMEhistoricalparent; parentUTCweekfold must beexcludedforTrainqueries.',
 source_calendar='Subtripdeparture timestamp=parenttimestamp+15*i; allsourceparentarrivalsstrictlybeforeValidation.',
 no_Validation_or_Test_rows_decoded=True,membership_checked_before_row_decode=True,source_endpoint_max_error=max_endpoint_error,
 no_persistent_pairbank=True,bank_MiB=file.stat().st_size/1024**2,outputs_sha256=sha(file),
 source_sha256=sha(folder/'fuentes'/Path(__file__).name),original_source_csv_sha256_recorded=manifest['source_sha256'],
 original_rows_labels_cohort_unchanged=True,Test=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(folder/'resultado.json',result);print(json.dumps(result),flush=True)
