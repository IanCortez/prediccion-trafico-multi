"""Outcome-free matched-K coverage test of additional historical Train parents."""
from base_replanteo import *
import csv,gc,shutil
from scipy.spatial import cKDTree
from numba import njit,prange,set_num_threads
from recuperar_subtrayectos import metric_xy,choose_subtrips
from datos_progreso_recurrente import lisbon_transitions

@njit(parallel=True,cache=True)
def support(oi,di,od,dd,parent_of_point,offset,xy):
 B,K=oi.shape;count=np.zeros(B,np.int32);error=np.full(B,np.inf,np.float64)
 for q in prange(B):
  op=parent_of_point[oi[q]];dp=parent_of_point[di[q]]
  so=np.argsort(op);sd=np.argsort(dp);a=0;b=0
  while a<K and b<K:
   po=op[so[a]];pd=dp[sd[b]]
   if po<pd:a+=1;continue
   if pd<po:b+=1;continue
   ae=a+1;be=b+1
   while ae<K and op[so[ae]]==po:ae+=1
   while be<K and dp[sd[be]]==po:be+=1
   best=np.inf
   for ia in range(a,ae):
    ao=so[ia];u=oi[q,ao];first=u-offset[po]
    for ib in range(b,be):
     ad=sd[ib];v=di[q,ad];last=v-offset[po];span=last-first
     if span<4 or span>119:continue
     dx=xy[v,0]-xy[u,0];dy=xy[v,1]-xy[u,1];length=np.sqrt(dx*dx+dy*dy)
     if length<200 or length/(15*span)*3.6>120:continue
     value=od[q,ao]**2+dd[q,ad]**2
     if value<best:best=value
   if np.isfinite(best):
    count[q]+=1
    if best<error[q]:error[q]=best
   a=ae;b=be
 return count,np.sqrt(error)

def main():
 access=deny_test();start=time.time();set_num_threads(2)
 run=OUT/'cobertura_fuentes_ampliadas_Train_s0';assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
 shutil.copy2(__file__,run/'fuentes'/Path(__file__).name)
 write(run/'plan.json',dict(scope='No-outcome coverage instrument, not MAE or a fitted candidate.',
  queries='512seeded rawOD/time heldTrain rows; labels/routes never read for scoring.',
  old='Original sourceparents with arrival strictlybefore first heldTrain query, selectedbeforeKDtree.',
  extra='Outside-primary rawparents, earlytimestamp and binaryN verify arrival before GPSdecode; anysourceintervalspan4..119.',
  comparison='Same2048nearestpoints perendpoint in old and combinedindex; no extraKbudget or radius tuning.',
  sources='All additional eligibleparents, including longer completejourneys; no sourceN enters query.',
  gate='Reduce zero-source queryfraction by>=10percentagepoints without reducing the fraction with a sourcewithin100m endpointRSSerror.',
  resources='All addedGPS RAM-only, no new pointbank on disk. OneCPU2 instrument alongside the publicfeaturebuilder.',
  limits='Support is not predictiveaccuracy and not an irreducibleerrorbound. No primaryValidation or Test accessed.'))
 # Long-parent synthetic test: absoluteindices>255 must remain correct, and span>119 must be rejected.
 xytest=np.column_stack([np.arange(300)*10.,np.zeros(300)]).astype(np.float32)
 c,e=support(np.array([[270]],np.int64),np.array([[290]],np.int64),np.zeros((1,1)),np.zeros((1,1)),np.zeros(300,np.int32),np.array([0,300]),xytest)
 assert c[0]==1 and e[0]==0
 c,e=support(np.array([[0]],np.int64),np.array([[299]],np.int64),np.zeros((1,1)),np.zeros((1,1)),np.zeros(300,np.int32),np.array([0,300]),xytest)
 assert c[0]==0 and np.isinf(e[0])
 tr=dict(np.load(DATA/'train_x.npz'));known=set(map(int,tr['row']))
 sourceq=OUT/'oraculo_tiempo_calle_train_s0/auxiliares_Train.npz'
 with np.load(sourceq) as z:q={k:z[k] for k in ['row','train_index','timestamp']}
 cutoff=int(q['timestamp'][40000:].min())
 qi=40000+np.sort(np.random.default_rng(20261124).choice(10000,512,replace=False))
 query={k:tr[k][q['train_index'][qi]] for k in ['o_lon','o_lat','d_lon','d_lat','timestamp']}
 qxy=metric_xy(query)
 manifest=read(DATA/'manifiesto.json');lat0,lat1,lon0,lon1=manifest['bbox']
 file=OUT/'memoria_puntos_Train_s0/puntos_Train.npz';assert sha(file)==read(file.parent/'resultado.json')['outputs_sha256']
 old=dict(np.load(file));oldparent=np.repeat(np.arange(len(old['parent_gps_n']),dtype=np.int32),old['parent_gps_n'])
 oldinside=np.unpackbits(old['indexable_packed'],count=len(oldparent)).astype(bool)
 oldeligible=(old['parent_gps_n']>=5)&(old['timestamp']+old['time_s']<cutoff)
 oldids=np.flatnonzero(oldinside&oldeligible[oldparent])
 tree=cKDTree(old['xy_m'][oldids].astype(np.float64))
 od,oi=tree.query(qxy[:,:2],k=2048,workers=2);dd,di=tree.query(qxy[:,2:],k=2048,workers=2)
 oldoi=oldids[oi];olddi=oldids[di]
 oldcount,olderror=support(oldoi,olddi,od,dd,oldparent,old['offsets'],old['xy_m'])
 # Original unmodified sampler is an independent actual old-bank availability check.
 starts,ends=lisbon_transitions()
 check=choose_subtrips(oldoi[:17],olddi[:17],od[:17],dd[:17],oldparent,old['offsets'],old['parent_gps_n'],old['timestamp'],old['parent_fold'],old['xy_m'],qxy[:17],query['timestamp'][:17],np.full(17,5),starts,ends)
 assert np.array_equal((check[0]>=0).any(1),oldcount[:17]>0)
 del tree,oldids,oldoi,olddi,od,dd,oi,di;gc.collect()
 print(json.dumps(dict(stage='old_support_complete',queries=512,seconds=time.time()-start)),flush=True)
 sx=111195*np.cos(np.deg2rad(41.16));sy=111195.
 chunks=[];nn=[];rows=[];timestamps=[]
 csv.field_size_limit(2**31-1)
 with (ROOT/'tesis/porto/porto/train.csv').open('rb') as f:
  next(f)
  for row,line in enumerate(f):
   if row in known:continue
   fields=line.split(b',',8);stamp=int(fields[5].strip(b'"'))
   if stamp>=cutoff-86400 or fields[7].strip(b'"')==b'True':continue
   binary_n=fields[8].count(b'[')-1
   if binary_n<5 or binary_n>5761:continue
   assert stamp+15*(binary_n-1)<cutoff
   values=next(csv.reader([line.decode('utf-8')]));points=np.asarray(json.loads(values[8]),np.float64);n=len(points)
   assert n==binary_n and np.isfinite(points).all()
   # Retain raworder/allpoints inRAM so intervalelapsedtime remains exact across outside-core excursions.
   xy=np.column_stack([(points[:,0]+8.62)*sx,(points[:,1]-41.16)*sy]).astype(np.float32)
   chunks.append(xy);nn.append(n);rows.append(row);timestamps.append(stamp)
 extra=np.concatenate(chunks);del chunks;gc.collect()
 sizes=np.r_[old['parent_gps_n'].astype(np.uint16),np.asarray(nn,np.uint16)]
 xy=np.concatenate([old['xy_m'],extra]);del extra
 parent=np.repeat(np.arange(len(sizes),dtype=np.int32),sizes);offset=np.r_[0,np.cumsum(sizes,dtype=np.int64)]
 timestamps=np.r_[old['timestamp'],np.asarray(timestamps,np.int64)]
 assert len(xy)==offset[-1] and len(parent)==len(xy)
 eligible=(timestamps+15*(sizes.astype(np.int64)-1)<cutoff)&(sizes>=5)
 bounds=old['bounds_m'];inside=(xy[:,0]>=bounds[0])&(xy[:,0]<=bounds[2])&(xy[:,1]>=bounds[1])&(xy[:,1]<=bounds[3])
 ids=np.flatnonzero(inside&eligible[parent]);del oldparent,oldinside,oldeligible,old;gc.collect()
 tree=cKDTree(xy[ids].astype(np.float64))
 od,oi=tree.query(qxy[:,:2],k=2048,workers=2);dd,di=tree.query(qxy[:,2:],k=2048,workers=2)
 count,error=support(ids[oi],ids[di],od,dd,parent,offset,xy)
 oldzero=oldcount==0;newzero=count==0
 before=dict(zero_source_fraction=float(oldzero.mean()),source_within100m_fraction=float((olderror<=100).mean()),median_source_count=float(np.median(oldcount)),finite_minerror_quantiles=np.quantile(olderror[np.isfinite(olderror)],[.1,.5,.9]).tolist())
 after=dict(zero_source_fraction=float(newzero.mean()),source_within100m_fraction=float((error<=100).mean()),median_source_count=float(np.median(count)),finite_minerror_quantiles=np.quantile(error[np.isfinite(error)],[.1,.5,.9]).tolist())
 gain=float(oldzero.mean()-newzero.mean());passed=gain>=.10 and after['source_within100m_fraction']>=before['source_within100m_fraction']
 np.savez_compressed(run/'cobertura_consultas_Train.npz',row=q['row'][qi],timestamp=query['timestamp'],old_source_count=oldcount,new_source_count=count,old_minerror_m=olderror,new_minerror_m=error)
 result=dict(complete=True,passed=bool(passed),query_n=512,before=before,after=after,zero_source_reduction_percentage_points=100*gain,
  gained_queries=int(np.sum(oldzero&~newzero)),lost_queries=int(np.sum(~oldzero&newzero)),
  additional_parents=len(nn),additional_points=int(np.sum(nn)),additional_GPS_bytes_persisted=0,
  source_selection_before_GPS_decode=True,held_query_GT_labels_routes_not_used=True,
  sameK2048=True,original_sampler_availability_exact17=True,long_parent_index_and_span_proof=True,
  primaryValidation_access=False,Test_access=False,source_sha256=sha(run/'fuentes'/Path(__file__).name),
  predictions_sha256=sha(run/'cobertura_consultas_Train.npz'),actual_data_files_accessed=sorted(access),seconds=time.time()-start,
  next_if_fail='Do not persist or train an expanded retrievalbank basedsolely on the amountofadditionaldata.')
 write(run/'resultado.json',result);print(json.dumps(read(run/'resultado.json')),flush=True)
if __name__=='__main__':main()
