"""Raw65publicinterface and kineticproxy audit on the same4096fitparents only."""
from base_replanteo import *
from inferencia import feature_input
from datos_mezcla_individual_Train import build_public206
from progreso_ruta_publica import public_polyline,project_fit_positions
from puente_temporal_estudiante import accumulated_ar_scale
import torch
start=time.time();access=deny_test()
run=OUT/'verificacion_progreso_publico_Train_s0';assert not run.exists();run.mkdir()
source=OUT/'auditoria_cinematica_ajuste_Train_s0'
audit=read(source/'resultado.json');assert audit['complete'] and audit['all_sampled_rows_in_original951718fit']
with np.load(source/'padres_ajuste_diagnostico.npz') as z:
 parent=z['parent'];rows=z['row'];targets=z['source_time_s'];timestamps=z['timestamp']
d,X,names=feature_input('train',read(BEST/'resultado.json'))
keep=np.array([not name.startswith('mm_') for name in names]);names=[n for n,k in zip(names,keep) if k]
X=np.ascontiguousarray(X[parent][:,keep]);assert len(names)==206
assert np.array_equal(d['row'][parent],rows) and np.array_equal(d['timestamp'][parent],timestamps)
q={k:d[k][parent] for k in ['o_lon','o_lat','d_lon','d_lat','timestamp']}
path,valid=public_polyline(q,X,names)
items=[{k:int(v[i]) if k=='timestamp' else float(v[i]) for k,v in q.items()} for i in range(65)]
rawX,rawnames=build_public206(items,read(BEST/'resultado.json')['config'])
assert rawnames==names and np.array_equal(rawX,X[:65])
rawq={k:v[:65] for k,v in q.items()};rawpath,rawvalid=public_polyline(rawq,rawX,names)
assert np.array_equal(rawpath,path[:65]) and np.array_equal(rawvalid,valid[:65])
reverse,_=public_polyline({k:v[::-1] for k,v in rawq.items()},rawX[::-1],names)
single,_=public_polyline({k:v[:1] for k,v in rawq.items()},rawX[:1],names)
assert np.array_equal(rawpath,reverse[::-1]) and np.array_equal(rawpath[:1],single)
# Independent straightline fixture has analytic normalizedprogress and rigid-transform invariance.
line=np.column_stack([np.linspace(10,1010,9),np.full(9,30.)])
xy=np.column_stack([np.linspace(10,1010,17),np.full(17,50.)])
coded,error=project_fit_positions(xy,line)
assert np.max(np.abs(coded.astype(float)/65535-np.linspace(0,1,17)))<=.5/65535+1e-15
assert np.max(np.abs(error-20))<1e-10
angle=.63;rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
coded2,error2=project_fit_positions(xy@rotation+123,line@rotation+123)
assert np.max(np.abs(coded.astype(int)-coded2.astype(int)))<=1 and np.allclose(error,error2,atol=1e-9)
# Signed AR covariance remains valid for negativecorrelation as well.
for rho in [-.95,-.4]:
 n=119;ix=np.arange(n);reference=(rho**np.abs(ix[:,None]-ix[None,:])).sum()
 actual=accumulated_ar_scale(torch.tensor([[rho]],dtype=torch.float64))[0,0,-1].item()
 assert abs(reference-actual)<1e-8
with np.load(OUT/'memoria_puntos_Train_s0/puntos_Train.npz') as z:
 offset=z['offsets'];gps_n=z['parent_gps_n']
 assert np.array_equal((gps_n[parent].astype(int)-1)*15,targets)
 assert np.all(timestamps+targets<1396531073)
 xy=z['xy_m']
 progress=np.zeros((4096,120),np.uint16);counts=gps_n[parent].astype(np.uint8)
 errors=[];correlations=[];negative=[];stationary=[];quantization_error=0.
 for i,p in enumerate(parent):
  points=xy[offset[p]:offset[p+1]]
  encoded,lateral=project_fit_positions(points,path[i]);n=len(encoded)-1
  assert 15*n==targets[i] and encoded[0]==0 and encoded[-1]==65535
  progress[i,:n+1]=encoded;values=encoded.astype(float)/65535;increments=np.diff(values)
  assert abs(increments.sum()-1)<1e-12
  centered=increments-increments.mean()
  den=np.sqrt((centered[:-1]**2).sum()*(centered[1:]**2).sum())
  if den>1e-12:correlations.append(float((centered[:-1]*centered[1:]).sum()/den))
  errors.extend(lateral.tolist());negative.append(float((increments<0).mean()));stationary.append(float((increments==0).mean()))
 del xy
file=run/'progreso_ajuste4096.npz'
np.savez_compressed(file,parent=parent,row=rows,timestamp=timestamps,private_fit_time_s=targets,
 private_fit_gps_n=counts,private_fit_progress_uint16=progress,public_route_available=valid)
quant=lambda a:np.quantile(np.asarray(a),[0,.1,.5,.9,.99,1]).tolist()
proof=dict(passed=True,scope='4096fit-targetrepresentation only; not predictiveMAE.',
 fit_n=4096,heldtrajectory_used=False,raw65public206_and_ninepoint_path_exact=True,
 raw_reverse65_singleton_exact=True,straightline_analytic_and_rigid_transform_passed=True,
 negative_AR_covariance_valid=True,
 all_original_durationlabels_preserved=True,all_fitparents_before1396531073=True,
 public_route_available_fraction=float(valid.mean()),fallback='Publicavailability-only straightline; no droppedqueries.',
 lateral_error_m_quantiles_p0_p10_p50_p90_p99_max=quant(errors),
 pertrip_increment_autocorrelation_quantiles=quant(correlations),
 mean_fraction_negative_progress_increment=float(np.mean(negative)),
 mean_fraction_exact_stationary_projected_increment=float(np.mean(stationary)),
 uint16_max_quantization_error=0.5/65535,
 interpretation='Signed publicrouteprogress is a proxy and can backtrack or pause due to routeambiguity/GPSnoise. Never equate it with exact physicalspeed.',
 source_target_archive_sha256=sha(file),target_archive_MiB=file.stat().st_size/1024**2,
 sources={n:sha(OUT/n) for n in ['verificar_progreso_publico_Train.py','progreso_ruta_publica.py']},
 Test=False,Validation=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(run/'resultado.json',proof)
for name in proof['sources']:(run/name).write_bytes((OUT/name).read_bytes())
print(json.dumps(proof))
