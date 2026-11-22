"""Extract low displacement TRAIN targets only; no trajectory bank is written."""
from base_replanteo import *
import csv

def main():
 start=time.time();access=deny_test();target=OUT/'descomposicion_train.npz';assert not target.exists();d=dict(np.load(DATA/'train_x.npz'));y=np.load(DATA/'train_y.npy');manifest=read(DATA/'manifiesto.json');cut=manifest['validation']['start_timestamp'];lookup=np.full(1710670,-1,np.int32);lookup[d['row']]=np.arange(len(y));low=np.zeros(len(y),np.uint8);reliable=np.zeros(len(y),bool);seen=np.zeros(len(y),bool);maxstep=np.zeros(len(y),np.float32);csv.field_size_limit(2**31-1);n=0
 with (ROOT/'tesis/porto/porto/train.csv').open(encoding='utf-8-sig',newline='') as f:
  for row,r in enumerate(csv.DictReader(f)):
   j=lookup[row]
   if j<0:continue
   p=np.asarray(json.loads(r['POLYLINE']),dtype=np.float64);assert (len(p)-1)*15==y[j];seen[j]=True
   if d['timestamp'][j]+y[j]>=cut:continue
   delta=np.diff(p,axis=0);delta[:,0]*=111195*np.cos(np.deg2rad(41.16));delta[:,1]*=111195;dist=np.sqrt((delta**2).sum(1));low[j]=np.count_nonzero(dist<5.);maxstep[j]=dist.max();reliable[j]=dist.max()<=500;n+=1
   if n%200000==0:print(json.dumps(dict(train_trips=n,seconds=time.time()-start)),flush=True)
 assert seen.all() and np.all(low<=y/15);move=(y/15).astype(np.uint8)-low;assert np.array_equal((move.astype(int)+low)*15,y)
 np.savez_compressed(target,row=d['row'],low_n=low,move_n=move,reliable=reliable,maxstep_m=maxstep)
 result=dict(n=len(y),reliable_n=int(reliable.sum()),embargoed_n=int(((d['timestamp']+y)>=cut).sum()),threshold_m=5.,reliable_rule='maximum 15-second step <=500m and arrival before first Validation departure',counts_are='Operational low-displacement GPS proxy; not annotated idling time',moving_plus_low_equals_total=True,query_trajectory_used=False,validation_or_test_path_decoded=False,source_csv_sha256_manifest=manifest['source_sha256'],output_sha256=sha(target),data_files_accessed=sorted(access),seconds=time.time()-start,code_sha256=sha(__file__))
 write(OUT/'descomposicion_manifiesto.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
