"""Train-only crossfit road-traffic tensor gate. Never use held TT to fit local field."""
from base_replanteo import *
import torch,pandas as pd,shutil,argparse,gc
from catboost import CatBoostRegressor
from inferencia import feature_input
from modelo_trafico_tensor import RoadTrafficTensor
ap=argparse.ArgumentParser();ap.add_argument('--departure-control',action='store_true');args=ap.parse_args()
access=deny_test();start=time.time();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction(.25)
run=OUT/('control_reloj_trafico_tensor_train_s0' if args.departure_control else 'piloto_trafico_tensor_train_s0');assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
for name in ['piloto_trafico_tensor_train.py','modelo_trafico_tensor.py','base_replanteo.py']:shutil.copy2(OUT/name,run/'fuentes'/name)
if args.departure_control:assert read(OUT/'piloto_trafico_tensor_train_s0/resultado.json')['passed']
meta=read(OUT/'datos_trafico_tensor_train_s0/resultado.json');file=OUT/'datos_trafico_tensor_train_s0/observaciones_Train.npz';assert sha(file)==meta['observations_sha256']
obs=dict(np.load(file));g=dict(np.load(OUT/'banco_valor_vial_s0/grafo_estatico.npz'));comp=dict(np.load(DATA/'grafo_compacto_2014.npz'));nroad=len(g['static'])
plan=dict(scope='Chronological heldTrain raw5ODfeature gate. Not primaryValidation or independentconfirmation.',
 mode='parent_departure_clock_control' if args.departure_control else 'observed_road_entry_clock',
 seed=20261122,source_observations=1000000,source_cutoff=meta['chronological_cutoff'],views='Six independentlytrainedfields: view0..4 excludesENTIREparentUTCweekfold inALLparameters/support/backoffs; heldview5usesallsourcesbeforecutoff.',
 local_target='Sum-preservingTrainpartialroadtime; positivepace*partialdistance+positivedwell, no actualqueryroute required. LocalclockatobservedTrainroadentry, not initialtripdeparture.',
 architecture='Rank16roadxweeklyquarterhour pace/dwell, publicstatic12background, supportshrinkcount/(count+10). Periodicslot672, learnableclockbias.',
 fit=dict(epochs=8,batch=8192,lr=.003,weight_decay=.0001,clip=5,clock_smoothness=.001,loss='Huberlog1ptime beta.5'),
 gate_features='16weightedmean/stdpace+dwell features over4publiccandidate roads at2endpoints, atdeparture andpublicfreeflowarrival. No truequeryN/road/duration.',
 primary_fit=200000,held=50000,control='Exactstored206numeric800tree control fromfineclock experiment; samefit/heldrows/params.',
 gate='At least1second heldTrainMAEimprovement andupperdaybootstrapCI<0. Onlythen matchedparentclockcontrol andfurtherprimarydesign.',
 storage='Allmetrics/predictions;6fields temporarilyneededforfeatureconstruction, keep onlyifgatepasses; <=120MiBnewartifacts.',
 Validation=False,Test=False)
write(run/'plan.json',plan)
d,X,names=feature_input('train',read(BEST/'resultado.json'));keep=np.asarray([not n.startswith('mm_') for n in names]);assert keep.sum()==206
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index'];y=np.load(DATA/'train_y.npy');ts=d['timestamp'].astype(np.int64);split=int(ts[eligible].max())-21*86400;assert split==meta['chronological_cutoff']
fp=eligible[ts[eligible]+y[eligible]<split];hp=eligible[ts[eligible]>=split];rng=np.random.default_rng(20261122);fit=np.sort(rng.choice(fp,200000,replace=False));held=np.sort(rng.choice(hp,50000,replace=False));indices=np.r_[fit,held];public=X[indices][:,keep].copy()
freeindex=names.index('network_fastest_time_600s');arrival=(ts[indices]+np.rint(X[indices,freeindex]*600).astype(np.int64));del X
prior=read(OUT/'piloto_reloj_fino_train_s0/resultado.json');ref=dict(np.load(OUT/'piloto_reloj_fino_train_s0/predicciones_held_Train.npz'))
assert hashlib.sha256(np.ascontiguousarray(d['row'][fit]).tobytes()).hexdigest()==prior['selected_fit_rows_sha256']
for key in ['row','timestamp']:assert np.array_equal(ref[key],d[key][held])
assert np.array_equal(ref['true'],y[held])
timestamp=ts[obs['train_index']] if args.departure_control else obs['entry_timestamp']
def slots(t):
 local=pd.to_datetime(t,unit='s',utc=True).tz_convert('Europe/Lisbon')
 return (np.asarray(local.dayofweek)*96+np.asarray(local.hour)*4+np.asarray(local.minute)//15).astype(np.int64)
obs_slot=slots(timestamp);assert obs_slot.min()>=0 and obs_slot.max()<672
road=torch.tensor(obs['chain'].astype(np.int64),device='cuda');slot=torch.tensor(obs_slot,device='cuda');distance=torch.tensor(obs['distance_m'],device='cuda');target=torch.tensor(obs['duration_s'],device='cuda')
fold=obs['parent_fold'];qfold=np.r_[(ts[fit]//604800)%5,np.full(len(held),5)];bankfile=OUT/'banco_valor_vial_s0/train_proyecciones.npz'
with np.load(bankfile) as z:bank={key:z[key][indices] for key in ['chain','error_m']}
qslots=np.column_stack([slots(ts[indices]),slots(arrival)]);features=np.zeros((len(indices),16),np.float32);models=[];histories=[];view_seeds=[]
for view in range(6):
 allowed=np.flatnonzero(fold!=view) if view<5 else np.arange(len(fold));assert view==5 or np.all(fold[allowed]!=view)
 support=np.bincount(obs['chain'][allowed],minlength=nroad);torch.manual_seed(20261122);torch.cuda.manual_seed_all(20261122)
 basepace=comp['chain_free']/np.maximum(comp['chain_length'],.001);model=RoadTrafficTensor(g['static'],basepace,support).cuda()
 opt=torch.optim.AdamW(model.parameters(),lr=plan['fit']['lr'],weight_decay=plan['fit']['weight_decay']);allowed_gpu=torch.tensor(allowed,device='cuda');history=[]
 for epoch in range(1,9):
  model.train();order=allowed_gpu[torch.randperm(len(allowed_gpu),device='cuda',generator=torch.Generator(device='cuda').manual_seed(7000+epoch))]
  total=0.;count=0
  for ix in order.split(8192):
   opt.zero_grad(set_to_none=True);pred=model(road[ix],slot[ix],distance[ix]);loss=torch.nn.functional.smooth_l1_loss(torch.log1p(pred),torch.log1p(target[ix]),beta=.5)+.001*model.smoothness()
   assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step();total+=float(loss.detach())*len(ix);count+=len(ix)
  history.append(dict(epoch=epoch,n=count,loss=total/count))
 model.eval();path=run/('campo_'+str(view)+'.pt');torch.save(dict(state=model.state_dict(),view=view,allowed_parent_folds=np.unique(fold[allowed]).tolist(),config=plan),path)
 models.append(dict(view=view,file=path.name,sha256=sha(path),observations=len(allowed),parent_folds=np.unique(fold[allowed]).tolist()));histories.append(dict(view=view,history=history))
 queries=np.flatnonzero(qfold==view)
 with torch.no_grad():
  for a in range(0,len(queries),2048):
   ix=queries[a:a+2048];chains=bank['chain'][ix].astype(np.int64);error=bank['error_m'][ix].astype(np.float64);prior_weight=np.exp(-.5*(error/30)**2);prior_weight/=np.maximum(prior_weight.sum(-1,keepdims=True),1e-20)
   ff=[]
   for time_index in [0,1]:
    road_ids=chains.reshape(-1);time_ids=np.broadcast_to(qslots[ix,time_index,None,None],chains.shape).reshape(-1)
    pace,dwell=model.components(torch.tensor(road_ids,device='cuda'),torch.tensor(time_ids,device='cuda'))
    values=np.stack([pace.cpu().numpy()*100,dwell.cpu().numpy()],-1).reshape(-1,2,4,2)
    mean=np.einsum('bek,bekf->bef',prior_weight,values);var=np.einsum('bek,bekf->bef',prior_weight,(values-mean[:,:,None,:])**2)
    ff.extend([mean.reshape(-1,4),np.sqrt(np.maximum(var,0)).reshape(-1,4)])
   features[ix]=np.concatenate(ff,1).astype(np.float16).astype(np.float32)
 print(json.dumps(dict(stage='crossfit_traffic_field',view=view,source_n=len(allowed),query_n=len(queries),final_source_loss=history[-1]['loss'],seconds=time.time()-start)),flush=True)
 del model,opt,allowed_gpu;gc.collect();torch.cuda.empty_cache()
write(run/'historia_campos.json',histories);write(run/'campos.json',models)
np.savez_compressed(run/'caracteristicas_Train.npz',row=d['row'][indices],timestamp=ts[indices],features=features,source_view=qfold)
assert np.isfinite(features).all()
params=dict(prior['plan']['model']);tte=CatBoostRegressor(**params);aug=np.concatenate([public,features],1)
tte.fit(aug[:200000],y[fit],verbose=False);pred=tte.predict(aug[200000:]);comparison=paired(y[held],pred,ref['control'],ts[held]);comparison['interpretation']='ChronologicalheldTrain only, sourcefieldparentweekcrossfit, everyparameterexcludesownsourcefold. ExtraTrainauxlabelsupervision not a matchedcapacitycausaltest.'
passed=bool(comparison['delta_mae_s']<=-1 and comparison['ci95_s'][1]<0)
if passed:tte.save_model(str(run/'tiempo_con_tensor.cbm'))
np.savez_compressed(run/'predicciones_held_Train.npz',row=d['row'][held],timestamp=ts[held],true=y[held],pred=pred,control=ref['control'])
result=dict(passed=passed,metrics=metric(y[held],pred),control=metric(y[held],ref['control']),comparison=comparison,scope=plan['scope'],
 local_fields=models,local_histories=histories,source_observations=meta['observations'],feature_count=16,
 all_source_arrivals_before_held_cutoff=True,all_own_parentfold_parameters_and_support_excluded=True,
 source_sha256=sha(run/'fuentes/piloto_trafico_tensor_train.py'),model_source_sha256=sha(run/'fuentes/modelo_trafico_tensor.py'),
 predictions_sha256=sha(run/'predicciones_held_Train.npz'),features_sha256=sha(run/'caracteristicas_Train.npz'),
 Validation_access=False,Test_access=False,seconds=time.time()-start)
write(run/'resultado.json',result)
if not passed:
 removed=[]
 for field in models:
  path=run/field['file'];assert path.resolve().is_relative_to(run.resolve()) and sha(path)==field['sha256'];removed.append(dict(file=path.name,sha256=field['sha256'],bytes=path.stat().st_size));path.unlink()
 write(run/'pesos_descartados.json',dict(reason='Gatefailed; metrics/predictions/sources preserved.',removed=removed))
assert sum(p.stat().st_size for p in run.rglob('*') if p.is_file())<120*1024**2
print(json.dumps(dict(passed=passed,metrics=result['metrics'],control=result['control'],delta_mae_s=comparison['delta_mae_s'],seconds=result['seconds'])),flush=True)
