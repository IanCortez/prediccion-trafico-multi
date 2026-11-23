"""Train-only diagnostic: information lost by compressing64 historical trips into6 prototypes."""
from base_replanteo import *
import shutil,pandas as pd
from scipy.spatial import cKDTree
from numba import njit,set_num_threads
from historicos_rutas import query_xy
access=deny_test();start=time.time();set_num_threads(2);run=OUT/'piloto_compresion_historial_s0'
assert not run.exists();run.mkdir();shutil.copy2(__file__,run/'fuente_ejecutada.py')
protocol=dict(scope='Train-only privileged GEOMETRY coverage instrument, not predictive MAE.',
 sample='Same20000Train queries as prior route/prototype gates.',
 query=['o_lon','o_lat','d_lon','d_lat','timestamp'],
 retrieval='Same nearest64 in4DOD, sourceUTCweekfold excluded,250mGaussian,4hourclock,width.35weekendmismatch,500mcutoff.',
 comparison='Public2routes plus straight and all available64reanchored historical shapes, versus previous6prototypes plus same public/straight fallback.',
 candidate_count_matched=False,reason='Diagnose compression loss; an oracle improves monotonically as hypotheses increase. Does not prove a learned attention model can select the right history.',
 gate='At least10meters AND15percent reduction in mean geometry oracle error versus6prototype compression; otherwise do not expand this memory context.',
 query_geometry_used_only_for_post_retrieval_scoring=True,query_time_label_used=False,
 Validation=False,Test='sealed',resources='CPU2,20kquery only; prediction/provenance archive <12MiB.')
write(run/'protocolo.json',protocol)
memory=OUT/'datos_prototipos_historicos_s0/memoria_Train.npz';meta=read(OUT/'datos_prototipos_historicos_s0/resultado.json')
assert sha(memory)==meta['memory']['sha256'];m=dict(np.load(memory))
oldpath=OUT/'piloto_prototipos_historicos_s0/predicciones_oraculo_Train.npz';old=dict(np.load(oldpath))
prior=dict(np.load(OUT/'piloto_proyeccion_incierta_s0_v2/predicciones_oraculo_Train.npz'))
assert np.array_equal(old['row'],prior['row']) and len(old['row'])==20000
tr=dict(np.load(DATA/'train_x.npz'));lookup=np.full(1710670,-1,np.int32);lookup[tr['row']]=np.arange(len(tr['row']))
qi=lookup[old['row']];assert (qi>=0).all() and np.array_equal(tr['timestamp'][qi],old['timestamp'])
d={k:tr[k][qi] for k in ['o_lon','o_lat','d_lon','d_lat','timestamp']};xy=query_xy(d);fold=(d['timestamp']//604800)%5
local=pd.to_datetime(d['timestamp'],unit='s',utc=True).tz_convert('Europe/Lisbon');hour=np.asarray(local.hour)+np.asarray(local.minute)/60+np.asarray(local.second)/3600;weekend=np.asarray(local.dayofweek)>=5
neighbors=np.empty((len(qi),64),np.int32);available=np.empty((len(qi),64),bool)
for f in range(5):
 take=np.flatnonzero(fold==f);source=np.flatnonzero(m['fold']!=f);tree=cKDTree(m['xy'][source])
 distance,index=tree.query(xy[take],k=64,workers=2);near=source[index]
 clock=np.abs(m['hour'][near]-hour[take,None]);clock=np.minimum(clock,24-clock)
 weight=np.exp(-.5*(distance/.25)**2-.5*(clock/4)**2)*np.where(m['weekend'][near]==weekend[take,None],1.,.35);weight[distance>.5]=0
 neighbors[take]=near;available[take]=weight>0
 assert (m['fold'][near]!=f).all() and not (m['row'][near]==d['row'][take,None]).any() if 'row' in d else (m['fold'][near]!=f).all()
 print(json.dumps(dict(stage='retrieve_without_query_geometry',fold=f,rows=len(take),seconds=time.time()-start)),flush=True)
# Query geometry is first consulted after the complete candidate set is frozen.
gt=dict(np.load(DATA/'geometria_train.npz'));pos=np.searchsorted(gt['row'],old['row']);assert np.array_equal(gt['row'][pos],old['row']);truth=gt['X'][pos,1:15]
@njit(cache=True)
def errors(xy,mxy,mg,neighbor,available,truth):
 n,K=neighbor.shape;err=np.full((n,K),np.inf,np.float32)
 for i in range(n):
  dx=xy[i,2]-xy[i,0];dy=xy[i,3]-xy[i,1];norm=max(np.sqrt(dx*dx+dy*dy),1e-9)
  for j in range(K):
   if not available[i,j]:continue
   k=neighbor[i,j];ox=mxy[k,2]-mxy[k,0];oy=mxy[k,3]-mxy[k,1];onorm=max(np.sqrt(ox*ox+oy*oy),1e-9)
   co=(dx*ox+dy*oy)/(norm*onorm);si=(dy*ox-dx*oy)/(norm*onorm);scale=norm/onorm;error=0.
   for p in range(7):
    gx=scale*(co*mg[k,2*p]-si*mg[k,2*p+1]);gy=scale*(si*mg[k,2*p]+co*mg[k,2*p+1])
    error+=np.sqrt((gx-truth[i,2*p])**2+(gy-truth[i,2*p+1])**2)/7*1000
   err[i,j]=error
 return err
err=errors(xy,m['xy'],m['geometry'],neighbors,available,truth)
expanded=np.minimum(prior['geometry_baseline_m'],err.min(1));compressed=old['historical8_geometry_m']
assert np.all(expanded<=compressed+.002)
gain=float(np.mean(compressed-expanded));fraction=gain/float(compressed.mean());passed=gain>=10 and fraction>=.15
np.savez_compressed(run/'diagnostico_geometria_Train.npz',row=old['row'],timestamp=old['timestamp'],
 compressed6_geometry_oracle_m=compressed,raw64_geometry_oracle_m=expanded,historical_source_rows=m['row'][neighbors],
 available=available,oracle_best_source_index=err.argmin(1))
size=(run/'diagnostico_geometria_Train.npz').stat().st_size;assert size<12*1024**2
result=dict(passed=bool(passed),n=len(qi),compressed6_mean_m=float(compressed.mean()),raw64_mean_m=float(expanded.mean()),
 mean_reduction_m=gain,relative_reduction=fraction,source_fold_exclusion=True,
 retrieval_frozen_before_query_geometry=True,query_true_time_used=False,
 candidate_count_matched=False,predictive_MAE_gain_not_established=True,
 conclusion='Gate passed: justify individual-history relational attention, with a learned-vs-matched-control full primary test.' if passed else 'Gate failed: do not expand to64history tokens just for geometry coverage.',
 source_sha256=sha(run/'fuente_ejecutada.py'),memory_sha256=sha(memory),previous_predictions_sha256=sha(oldpath),
 diagnostic_predictions_sha256=sha(run/'diagnostico_geometria_Train.npz'),diagnostic_bytes=size,
 Validation_access=False,Test_access=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(run/'resultado.json',result);print(json.dumps({k:result[k] for k in ['passed','n','compressed6_mean_m','raw64_mean_m','mean_reduction_m','relative_reduction','seconds']}),flush=True)
