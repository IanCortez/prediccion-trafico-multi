"""Independent NumPy source geometry/time/context reconstruction on actual Train intervals."""
from base_replanteo import *
import torch,pandas as pd
from tokens_subtrayectos import SubtripTokenBank,query_parts_sub
access=deny_test();start=time.time();torch.set_num_threads(2)
d=dict(np.load(DATA/'train_x.npz'));idx=dict(np.load(OUT/'indices_subtrayectos_s0/train_indices.npz'))
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index']
ii=np.sort(np.random.default_rng(677).choice(eligible,128,replace=False));part=query_parts_sub(d);bank=SubtripTokenBank('cpu')
args=[torch.as_tensor(part[k][ii]) for k in ['xy_m','hour','weekend']]
actual=bank.features(*args,idx['parent'][ii],idx['first'][ii],idx['last'][ii]).float().numpy()
raw=np.zeros_like(actual,dtype=np.float64);rawcoords=bank.xy.numpy();offsets=bank.offset.numpy();ns=bank.n.numpy();timestamps=bank.ts.numpy()
fractions=np.arange(1,8)/8
for q,index in enumerate(ii):
 qxy=part['xy_m'][index].astype(float);qd=qxy[2:]-qxy[:2];qn=np.linalg.norm(qd)
 for k,parent in enumerate(idx['parent'][index]):
  if parent<0:continue
  first=int(idx['first'][index,k]);last=int(idx['last'][index,k]);points=rawcoords[offsets[parent]+np.arange(first,last+1)].astype(float)
  source=np.r_[points[0],points[-1]];sd=points[-1]-points[0];sn=np.linalg.norm(sd);scale=np.clip(qn/sn,.5,2)
  co=np.dot(qd,sd)/(qn*sn);si=(qd[1]*sd[0]-qd[0]*sd[1])/(qn*sn)
  middle=np.column_stack([np.interp(fractions*(last-first),np.arange(len(points)),points[:,a]) for a in range(2)])
  residual=middle-points[0]-fractions[:,None]*sd
  rot=np.array([[co,-si],[si,co]]);shape=(residual@rot.T*scale/1000).reshape(-1)
  poly=np.vstack([points[0],middle,points[-1]]);path=np.linalg.norm(np.diff(poly,axis=0),axis=1).sum()*scale/1000
  delta=(source-qxy)/1000;distance=np.linalg.norm(delta)
  local=pd.Timestamp(int(timestamps[parent])+15*first,unit='s',tz='UTC').tz_convert('Europe/Lisbon')
  sh=local.hour+local.minute/60+local.second/3600
  clock=(sh-float(part['hour'][index])+12)%24-12;phase=clock*2*np.pi/24;match=float((local.dayofweek>=5)==part['weekend'][index])
  prior=-.5*(distance/.05)**2-.5*(clock/4)**2+(0 if match else np.log(.35));den=max(1,int(ns[parent])-1)
  raw[q,k]=np.r_[shape,path,np.log(max(.01,path/(qn/1000))),np.log(max(.1,15*(last-first)*scale)),
   delta,distance,np.sin(phase),np.cos(phase),match,np.log(scale),prior,1,
   first/den,(int(ns[parent])-1-last)/den,float(first==0),float(last==int(ns[parent])-1),np.log(den*15),(last-first)/den]
valid=idx['parent'][ii]>=0
error=np.abs(actual.astype(float)-raw);scaled=error/(1+np.abs(raw))
assert np.isfinite(actual).all() and scaled.max()<.003,(scaled.max(),np.unravel_index(scaled.argmax(),scaled.shape))
assert np.array_equal(actual[:,:,27]>0,valid)
assert np.all(actual[:,:,28:32][valid]>=0) and np.all(actual[:,:,28:32][valid]<=1)
result=dict(passed=True,Train_queries=128,source_intervals=int(valid.sum()),columns=34,independent_numpy_calendar_geometry_time_context=True,
 max_relative_scaled_error=float(scaled.max()),percolumn_max_scaled_error=scaled.max((0,1)),
 tolerance=.003,reason='DerivedcanonicalFloat16tokens; NumPyreference calculates inFloat64.',
 query_fields=['o_lon','o_lat','d_lon','d_lat','timestamp'],no_query_target_read=True,
 source_sha256=sha(__file__),token_sha256=sha(OUT/'tokens_subtrayectos.py'),Test_access=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(OUT/'verificacion_geometria_tokens_subtrayectos.json',result)
print(json.dumps(dict(passed=True,intervals=int(valid.sum()),max_scaled_error=float(scaled.max()),seconds=result['seconds'])),flush=True)
