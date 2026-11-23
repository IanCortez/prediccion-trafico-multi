"""Same4096fitparents: exactlinearframe, labelidentity, rigidtransforms, GPSdiagnostics."""
from base_replanteo import *
from coordenadas_trayecto_2d import metric_queries,fit_positions
start=time.time();access=deny_test()
run=OUT/'verificacion_coordenadas_2d_Train_s0';assert not run.exists();run.mkdir()
with np.load(OUT/'auditoria_cinematica_ajuste_Train_s0/padres_ajuste_diagnostico.npz') as z:
 parents=z['parent'];row=z['row'];times=z['source_time_s'];timestamp=z['timestamp']
with np.load(OUT/'indices_escala_Train_s0/metadatos_Train.npz') as z:
 fit_indices=z['train_index'][:int(z['fit_n'])]
 assert np.isin(parents,fit_indices).all()
with np.load(DATA/'train_x.npz') as d:
 assert np.array_equal(d['row'][parents],row) and np.array_equal(d['timestamp'][parents],timestamp)
 q={key:d[key][parents] for key in ['o_lon','o_lat','d_lon','d_lat','timestamp']}
xy=metric_queries(q);assert np.isfinite(xy).all()
# Analytic independent fixture: rotatedrectangle with a known interiorpoint.
points=np.array([[10.,20.],[35.,50.],[70.,100.]])
frame=np.r_[points[0],points[-1]]
position,_=fit_positions(points,frame)
u=(points[-1]-points[0])/100
expected=np.array([[(points[1]-points[0])@u/100,np.cross(u,points[1]-points[0])/100]])
assert np.allclose(position[1:2],expected,atol=1e-14)
angle=.71;rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
transformed=points@rotation+1234
position2,_=fit_positions(transformed,np.r_[transformed[0],transformed[-1]])
assert np.allclose(position,position2,atol=1e-14)
values=np.zeros((4096,120,2),np.float32);counts=[];end_errors=[];corr=[];neg=[];maxabs=[];circuit=[]
with np.load(OUT/'memoria_puntos_Train_s0/puntos_Train.npz') as z:
 off=z['offsets'];sizes=z['parent_gps_n'];points=z['xy_m']
 assert np.array_equal(15*(sizes[parents].astype(int)-1),times) and np.all(timestamp+times<1396531073)
 for i,p in enumerate(parents):
  raw=points[off[p]:off[p+1]]
  position,error=fit_positions(raw,xy[i]);n=len(position)-1
  assert 15*n==times[i] and np.array_equal(position[0],[0,0]) and np.array_equal(position[-1],[1,0])
  values[i,:n+1]=position;counts.append(n);end_errors.append(error)
  increments=np.diff(position,axis=0);assert np.allclose(increments.sum(0),[1,0],atol=1e-12)
  centered=increments-increments.mean(0)
  den=np.sqrt((centered[:-1]**2).sum()*(centered[1:]**2).sum())
  if den>1e-12:corr.append(float((centered[:-1]*centered[1:]).sum()/den))
  neg.append(float((increments[:,0]<0).mean()));maxabs.append(float(np.abs(position).max()))
  circuit.append(float(np.sqrt((np.diff(raw.astype(float),axis=0)**2).sum(1)).sum()/np.linalg.norm(xy[i,2:]-xy[i,:2])))
 del points
file=run/'coordenadas_ajuste4096.npz'
np.savez_compressed(file,parent=parents,row=row,timestamp=timestamp,private_fit_n=np.array(counts,np.uint8),
 private_fit_positions=values,query_xy=xy)
quant=lambda a:np.quantile(a,[0,.1,.5,.9,.99,1]).tolist()
proof=dict(passed=True,scope='4096precutoffFITcoordinates only, no newpredictiveMAE.',
 fit_rows_preserved=True,true_duration_labels_preserved=True,no_heldGPS=True,all_fitparents_before1396531073=True,
 linearframe_independent_fixture_passed=True,rigidtransform_invariance=True,
 endpoint_mismatch_m_quantiles=quant(end_errors),max_normalized_position_quantiles=quant(maxabs),
 vector_increment_autocorrelation_quantiles=quant(corr),mean_negative_longitudinal_fraction=float(np.mean(neg)),
 GPS_path_circuity_quantiles=quant(circuit),
 no_nearest_publicpath_projection=True,no_big_populationcache=True,
 interpretation='Signed2DprojectedGPS motion includes roadturns, waiting and GPSerrors; globalStudent scale does not identify their separatephysicalcauses.',
 data_sha256=sha(file),data_MiB=file.stat().st_size/1024**2,
 sources={name:sha(OUT/name) for name in ['verificar_coordenadas_2d_Train.py','coordenadas_trayecto_2d.py']},
 Validation=False,Test=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(run/'resultado.json',proof)
for name in proof['sources']:(run/name).write_bytes((OUT/name).read_bytes())
print(json.dumps(proof))
