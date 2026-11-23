"""Chronological Train-only privileged-route diagnostic; never a primary OD score."""
from base_replanteo import *
import csv,ast,shutil,importlib.util
from numba import set_num_threads
from catboost import CatBoostRegressor
from inferencia import feature_input
from consulta_mapa2014 import resources,ORIGIN
from map_matching import project_candidates
import _trazador_viterbi_20261122 as trace_module
sys.modules['trazador_viterbi_20261122']=trace_module
viterbi_trace=trace_module.viterbi_trace
access=deny_test();start=time.time();set_num_threads(2)
run=OUT/'oraculo_tiempo_calle_train_s0';assert not (run/'resultado.json').exists() and not (run/'descriptores_ruta_Train.npz').exists();run.mkdir(exist_ok=True);shutil.copy2(__file__,run/'fuente_ejecutada.py')
proof=read(OUT/'verificacion_unite_neural_absorcion.json');assert proof['passed']
plan=dict(scope='Privileged actual-route chronological Train-only diagnostic; not deployable OD, primaryValidation or a lower bound.',
 hypothesis='Street geometry and traversal-specific distances might explain duration beyond rawOD/public features, enough to warrant a latent road-route model.',
 seed=20261122,fit_n=40000,held_n=10000,held_last_days=21,
 fit='Quality eligibleTrain memory only; all fit arrivals precede held cutoff. Auxiliary diagnostic selection never changes primary cohort.',
 route='Original frozen4candidate matcher, traced without changing output; trusted15secondinterval paths bridged, unknownGAPs masked. Only street geometry/distance enters oracle; not true duration, pertoken time, GPSsample count, reset count or unknown elapsed time.',
 features='Matched206public ODfeatures control; treatment adds known route distance/freeflow/geometry orientation8, speed bins6, distance-weighted hashed street64, start/end partial distances and geometric route count. Actualroute is privileged.',
 model=dict(iterations=1000,depth=7,learning_rate=.05,loss_function='MAE',l2_leaf_reg=5,random_seed=20261122,thread_count=2,allow_writing_files=False,verbose=False),
 stopping='Fixed1000trees; no held early stopping or hyperparameter search.',
 gate='At least3seconds reduction in heldTrain MAE and upper95percent daybootstrap deltaCI<0. Necessary diagnostic only, not evidence an OD generator can recover actual route.',
 resources='CPU2,1024trip chunks, no persistent rawGPS/candidate cache, <=200MiB artifacts.',
 Validation=False,Test=False)
write(run/'plan.json',plan)
m=dict(np.load(OUT/'datos_prototipos_historicos_s0/memoria_Train.npz'));ts=m['timestamp'].astype(np.int64);target=m['time_s']
# Same cutoff as prior chronological predictive gates, not dependent on chosen sample.
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index'];primary_ts=np.load(DATA/'train_queries.npz')['timestamp'] if (DATA/'train_queries.npz').exists() else None
rec=read(BEST/'resultado.json');d,X,names=feature_input('train',rec);keep=np.asarray([not n.startswith('mm_') for n in names]);assert keep.sum()==206
split=int(d['timestamp'][eligible].max())-21*86400
fit_pool=np.flatnonzero(ts+target<split);held_pool=np.flatnonzero(ts>=split);rng=np.random.default_rng(plan['seed'])
fit=np.sort(rng.choice(fit_pool,plan['fit_n'],replace=False));held=np.sort(rng.choice(held_pool,plan['held_n'],replace=False));selected=np.r_[fit,held];n=len(selected)
rows=m['row'][selected];train_index=m['train_index'][selected];assert np.array_equal(rows,d['row'][train_index])
public=X[train_index][:,keep].copy();del X,primary_ts
assert ts[fit].max()+0<split and np.all(ts[fit]+target[fit]<split) and np.all(ts[held]>=split)
raw,comp,sample_edges,tree,routes,profiles=resources();set_num_threads(2);D,conn,routing=routes['length']
flat=comp['flat_incoming_arc'];valid=flat>=0;pc=np.repeat(np.arange(len(comp['chain_src']),dtype=np.int32),np.diff(comp['chain_offsets']))
edge_chain=np.full(len(raw['u']),-1,np.int32);edge_chain[flat[valid]]=pc[valid];assert np.all(edge_chain>=0)
# Copy only exact helper definitions from executed, time-preserving pilot; importing that pilot would rerun it.
source=OUT/'preparar_piloto_cadenas_parciales.py';old=read(OUT/'piloto_cadenas_parciales_train_s0/resultado.json');assert sha(source)==old['source_sha256']
parsed=ast.parse(source.read_text(encoding='utf-8-sig'));functions=[q for q in parsed.body if isinstance(q,ast.FunctionDef) and q.name in ['append_token','one_trip']];assert len(functions)==2
module_path=run/'helpers_parciales.py';module_path.write_text('from base_replanteo import np\nfrom numba import njit\nfrom ruteo_proyectado import collect_cost_path\n'+''.join(ast.unparse(q)+'\n' for q in functions),encoding='utf-8')
spec=importlib.util.spec_from_file_location('oraculo_parciales_helpers',module_path);helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)
lookup={int(row):i for i,row in enumerate(rows)};route=np.empty((n,90),np.float32);seen=np.zeros(n,bool);codes=[None]*n;times=[None]*n;distances=[None]*n;masks=[None]*n
sx=comp['junction_raw'][comp['chain_src']];dx=comp['junction_raw'][comp['chain_dst']]
vector=raw['xy'][dx]-raw['xy'][sx];angle=(np.arctan2(vector[:,1],vector[:,0])+2*np.pi)%(2*np.pi);orientation=np.floor(angle*8/(2*np.pi)).astype(np.int32)%8
speed=comp['chain_length']/np.maximum(comp['chain_free'],.001);speedbin=np.digitize(speed,[4,7,10,14,20])
trusted_time=0.;total_time=0.;maximum_error=0.;token_count=0;chunks=0
def process(batch):
 global trusted_time,total_time,maximum_error,token_count,chunks
 indices=np.asarray([a for a,p in batch],np.int32);ll=np.vstack([p for a,p in batch]);offset=np.r_[0,np.cumsum([len(p) for a,p in batch])].astype(np.int64)
 points=np.column_stack([(ll[:,0]-ORIGIN[0])*111195*np.cos(np.deg2rad(ORIGIN[1])),(ll[:,1]-ORIGIN[1])*111195])
 _,near=tree.query(points,k=32,workers=2);edges,frac,error=project_candidates(points,near,sample_edges,raw['xy'],raw['u'],raw['v'])
 be,bf,ber,resets,flags=viterbi_trace(points,offset,edges,frac,error,raw['u'],raw['v'],raw['length'],D,*conn)
 for j,(a,b) in zip(indices,zip(offset[:-1],offset[1:])):
  c,t,dist,unknown,fails,stationary,overflow=helper.one_trip(a,b,be,bf,flags,ber,raw['u'],raw['v'],raw['length'],D,conn,routing,edge_chain)
  y=float(target[selected[j]]);err=abs(float(t.astype(np.float64).sum())-y);assert err<.001
  maximum_error=max(maximum_error,err);known=c!=65535;cc=c[known].astype(np.int32);dd=dist[known].astype(np.float64);total=float(dd.sum())
  free=float((dd*comp['chain_free'][cc]/np.maximum(comp['chain_length'][cc],.001)).sum())
  # No elapsed token times or sample count are included in predictive features.
  histo=np.bincount(orientation[cc],weights=dd,minlength=8)/max(total,1.)
  histv=np.bincount(speedbin[cc],weights=dd,minlength=6)/max(total,1.)
  hashed=np.bincount((cc.astype(np.int64)*2654435761%64).astype(np.int32),weights=dd,minlength=64)/max(total,1.)
  first=float(dd[0]) if len(dd) else 0.;last=float(dd[-1]) if len(dd) else 0.
  chord=float(np.linalg.norm(points[b-1]-points[a]))
  basic=np.array([total/1000,free/600,len(cc)/100,total/max(chord,1),first/1000,last/1000,
    float(np.sum(dd*comp['chain_length'][cc])/max(total,1))/1000,float(np.max(dd)) if len(dd) else 0,
    float(np.sum(dd*speed[cc])/max(total,1))/10,float(np.sum(dd*(orientation[cc]%2)))/max(total,1),
    float(np.sum(dd[cc==cc[0]]))/max(total,1) if len(cc) else 0,float(np.sum(dd[cc==cc[-1]]))/max(total,1) if len(cc) else 0],np.float64)
  value=np.r_[basic,histo,histv,hashed];assert len(value)==90 and np.isfinite(value).all();route[j]=value
  mask=np.zeros(len(c),bool)
  if len(c)>1:
   ix=np.flatnonzero(known[:-1]&known[1:]);mask[ix]=comp['chain_dst'][c[ix].astype(np.int32)]==comp['chain_src'][c[ix+1].astype(np.int32)]
  codes[j]=c;times[j]=t;distances[j]=dist;masks[j]=mask;seen[j]=True
  trusted_time+=float(t[known].astype(np.float64).sum());total_time+=y;token_count+=len(c)
 chunks+=1
 print(json.dumps(dict(stage='Train_route_diagnostic_mapping',processed=int(seen.sum()),tokens=token_count,seconds=time.time()-start)),flush=True)
 write(run/'progreso.json',dict(stage='mapping',processed=int(seen.sum()),target_n=n,tokens=token_count,seconds=time.time()-start))
batch=[]
with (ROOT/'tesis/porto/porto/train.csv').open('rb') as f:
 header=next(csv.reader([next(f).decode('utf-8-sig').strip()]));pcol=header.index('POLYLINE');tcol=header.index('TIMESTAMP')
 for rawrow,line in enumerate(f):
  j=lookup.get(rawrow)
  if j is None:continue
  fields=next(csv.reader([line.decode('utf-8')]));assert int(fields[tcol])==ts[selected[j]]
  p=np.asarray(json.loads(fields[pcol]),np.float64);assert (len(p)-1)*15==target[selected[j]]
  batch.append((j,p))
  if len(batch)>=1024:process(batch);batch=[]
  if seen.sum()+len(batch)==n:break
 if batch:process(batch)
assert seen.all()
ptr=np.r_[0,np.cumsum([len(c) for c in codes])].astype(np.int64)
np.savez_compressed(run/'auxiliares_Train.npz',row=rows,train_index=train_index,timestamp=ts[selected],true=target[selected],fit_n=plan['fit_n'],
 offsets=ptr,chain=np.concatenate(codes),duration_s=np.concatenate(times),distance_m=np.concatenate(distances),next_route_valid=np.concatenate(masks))
np.savez_compressed(run/'descriptores_ruta_Train.npz',row=rows,features=route)
assert sum(p.stat().st_size for p in run.iterdir() if p.is_file())<200*1024**2
print(json.dumps(dict(stage='fitting_matched_OD_control',seconds=time.time()-start)),flush=True)
nf=plan['fit_n'];y=target[selected].astype(np.float64);control=CatBoostRegressor(**plan['model']);control.fit(public[:nf],y[:nf]);p0=control.predict(public[nf:]);del control
print(json.dumps(dict(stage='fitting_privileged_route_diagnostic',control=metric(y[nf:],p0),seconds=time.time()-start)),flush=True)
augmented=np.concatenate([public,route],1);model=CatBoostRegressor(**plan['model']);model.fit(augmented[:nf],y[:nf]);p1=model.predict(augmented[nf:])
comparison=paired(y[nf:],p1,p0,ts[selected[nf:]]);comparison['interpretation']='Privileged-route chronological heldTrain diagnostic, not primary OD performance or bound.'
passed=bool(comparison['delta_mae_s']<=-3 and comparison['ci95_s'][1]<0)
if passed:model.save_model(str(run/'oraculo_ruta.cbm'))
np.savez_compressed(run/'predicciones_held_Train.npz',row=rows[nf:],timestamp=ts[selected[nf:]],true=y[nf:],control=p0,privileged_route=p1)
result=dict(passed=passed,scope=plan['scope'],metrics=dict(control=metric(y[nf:],p0),privileged_route=metric(y[nf:],p1)),comparison=comparison,
 trips=n,tokens=token_count,trusted_time_fraction=trusted_time/total_time,max_auxiliary_sum_error_s=maximum_error,
 feature_count=dict(control=public.shape[1],privileged_route=augmented.shape[1]),no_token_times_GPSsample_counts_or_labels_in_features=True,
 fit_arrival_purge=True,primary_cohort_unchanged=True,Validation_access=False,Test_access=False,
 source_sha256=sha(run/'fuente_ejecutada.py'),helper_sha256=sha(module_path),auxiliary_sha256=sha(run/'auxiliares_Train.npz'),
 predictions_sha256=sha(run/'predicciones_held_Train.npz'),actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(run/'resultado.json',result);print(json.dumps(result),flush=True)
