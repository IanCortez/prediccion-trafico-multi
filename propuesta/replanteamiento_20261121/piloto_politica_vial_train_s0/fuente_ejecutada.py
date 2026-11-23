"""Teacher-state route predictability diagnostic on chronological Train only, not an OD result."""
from base_replanteo import *
import shutil
from catboost import CatBoostClassifier
from consulta_mapa2014 import resources
from numba import set_num_threads
access=deny_test();start=time.time();set_num_threads(2);run=OUT/'piloto_politica_vial_train_s0';assert not run.exists();run.mkdir();shutil.copy2(__file__,run/'fuente_ejecutada.py')
source=OUT/'oraculo_tiempo_calle_train_s0';meta=read(source/'resultado.json');assert meta['passed'];z=dict(np.load(source/'auxiliares_Train.npz'));assert sha(source/'auxiliares_Train.npz')==meta['auxiliary_sha256']
plan=dict(scope='Privileged known-current-road and true-terminal-road chronological Train route-predictability diagnostic. Not ODprediction or endtoendroute accuracy.',
 fit_trips=40000,held_trips=10000,max_states_per_trip=4,seed=20261122,
 input='Publicroad attributes, public directed shortest remainingdistance, turn angles, source/goalgeometry, departureclock. Actualcurrent/dest road are privileged diagnostic state only. No elapsedtime, futurelength, pertokenTT, driver, GPSpointcount.',
 control='Greedypublicshortestremainingdistance neighbor; EOSavailableonlywhen currentroad equals destinationroad, never from observedrouteN.',
 model=dict(iterations=400,depth=6,learning_rate=.05,l2_leaf_reg=5,loss_function='Logloss',random_seed=20261122,thread_count=2,allow_writing_files=False,verbose=False),
 stop='Fixed400trees,noheldstopping or search.',gate='At least3percentagepoints top1 improvement and upper95percentdaybootstrap deltaNLL<0. Prerequisiteonly; teacherstates do not establish ODroutegeneration quality.',
 resources='CPU2, <=200MiBartifacts.',Validation=False,Test=False)
write(run/'plan.json',plan)
raw,c,sample,tree,routes,profiles=resources();set_num_threads(2);D,conn,routing=routes['length'];g=dict(np.load(OUT/'banco_valor_vial_s0/grafo_estatico.npz'))
neighbors=g['neighbors'];nv=g['neighbor_valid'];deg=neighbors.shape[1];static=g['static'];nroad=len(static);length=c['chain_length'];free=c['chain_free']
sx=raw['xy'][c['junction_raw'][c['chain_src']]];dx=raw['xy'][c['junction_raw'][c['chain_dst']]];heading=(dx-sx)/np.maximum(np.linalg.norm(dx-sx,axis=1,keepdims=True),.001)
d=dict(np.load(DATA/'train_x.npz'));tridx=z['train_index'];rng=np.random.default_rng(plan['seed'])
features=[];labels=[];controls=[];group=[];query=[];state_fit=[];ptr=[0];gtchoice=[];times=[]
for i,(a,b) in enumerate(zip(z['offsets'][:-1],z['offsets'][1:])):
 destination=int(z['chain'][b-1]);origin=int(z['chain'][a])
 if destination==65535 or origin==65535:continue
 eligible=np.r_[np.flatnonzero(z['next_route_valid'][a:b])+a,b-1]
 eligible=eligible[z['chain'][eligible]!=65535]
 if not len(eligible):continue
 selected=rng.choice(eligible,min(4,len(eligible)),replace=False)
 for p in selected:
  cur=int(z['chain'][p]);actual=nroad if p==b-1 else int(z['chain'][p+1])
  candidates=neighbors[cur][nv[cur]].tolist()
  if cur==destination:candidates.append(nroad)
  if actual not in candidates:continue
  cand=np.asarray(candidates,np.int32);end=cand==nroad;safe=np.where(end,cur,cand);nxtdist=D[c['chain_dst'][safe],c['chain_src'][destination]]
  cost=length[safe]+nxtdist;cost=np.where(end,0,cost)
  # Candidate traversal of terminalchain ends directly, without needing its exit to loop back to its entry.
  cost=np.where(cand==destination,length[safe],cost);cost=np.where(np.isfinite(cost),cost,1e7)
  baseline=cost.copy();current_best=float(np.min(cost))
  turn=heading[safe,0]*heading[cur,1]-heading[safe,1]*heading[cur,0];cosine=(heading[safe]*heading[cur]).sum(1)
  goal=sx[destination]-dx[cur];goal=goal/max(float(np.linalg.norm(goal)),1.)
  source_vector=dx[cur]-sx[origin];source_norm=float(np.linalg.norm(source_vector))
  departure=float(d['timestamp'][tridx[i]]%86400)/86400*2*np.pi
  common=np.tile(np.r_[static[cur],static[destination],(sx[destination]-sx[origin])/10000,source_norm/10000,np.sin(departure),np.cos(departure)],(len(cand),1))
  values=np.column_stack([static[safe],end.astype(float),(cand==destination).astype(float),np.log1p(cost),np.log1p(np.maximum(cost-current_best,0)),turn,cosine,
   (heading[safe]*goal).sum(1),np.linalg.norm(dx[safe]-sx[destination],axis=1)/10000,common]).astype(np.float32)
  assert np.isfinite(values).all()
  features.append(values);labels.append((cand==actual).astype(np.int32));controls.append(baseline);group.append(cand);query.append(i);state_fit.append(i<plan['fit_trips'])
  ptr.append(ptr[-1]+len(cand));gtchoice.append(int(np.flatnonzero(cand==actual)[0]));times.append(z['timestamp'][i])
F=np.concatenate(features);Y=np.concatenate(labels);cost=np.concatenate(controls);ptr=np.asarray(ptr,np.int64);state_fit=np.asarray(state_fit,bool);times=np.asarray(times,np.int64)
row_fit=np.repeat(state_fit,np.diff(ptr));assert row_fit.any() and (~row_fit).any()
print(json.dumps(dict(stage='fitting_route_policy',fit_states=int(state_fit.sum()),held_states=int((~state_fit).sum()),candidate_rows=len(F),features=F.shape[1],seconds=time.time()-start)),flush=True)
model=CatBoostClassifier(**plan['model']);model.fit(F[row_fit],Y[row_fit]);held_flat=np.flatnonzero(~row_fit);score=model.predict(F[held_flat],prediction_type='RawFormulaVal');allscore=np.full(len(F),np.nan);allscore[held_flat]=score
chosen=[];greedy=[];true=[];nll=[];control_nll=[];held_ids=np.flatnonzero(~state_fit)
for i in held_ids:
 a,b=ptr[i:i+2];s=allscore[a:b];prob=np.exp(s-np.max(s));prob/=prob.sum();truth=int(gtchoice[i]);chosen.append(int(np.argmax(s)));greedy.append(int(np.argmin(cost[a:b])));true.append(truth)
 nll.append(-np.log(max(prob[truth],1e-12)))
 # Fixedpublicgeometrysoftmax has a temperature of250metres; no fit toheld labels.
 v=-cost[a:b]/250;v=np.exp(v-v.max());v/=v.sum();control_nll.append(-np.log(max(v[truth],1e-12)))
chosen=np.asarray(chosen);greedy=np.asarray(greedy);true=np.asarray(true);nll=np.asarray(nll);control_nll=np.asarray(control_nll);ht=times[held_ids]
days,ix=np.unique(ht//86400,return_inverse=True);ds=np.bincount(ix,weights=nll-control_nll);counts=np.bincount(ix);rr=np.random.default_rng(1729).integers(len(days),size=(3000,len(days)));ci=np.quantile(ds[rr].sum(1)/counts[rr].sum(1),[.025,.975])
accuracy=float(np.mean(chosen==true));base=float(np.mean(greedy==true));passed=bool(accuracy-base>=.03 and ci[1]<0)
if passed:model.save_model(str(run/'politica.cbm'))
np.savez_compressed(run/'predicciones_held_Train.npz',query_row=z['row'][np.asarray(query)[held_ids]],timestamp=ht,true_choice=true,choice=chosen,greedy=greedy,nll=nll,control_nll=control_nll)
result=dict(passed=passed,scope=plan['scope'],fit_states=int(state_fit.sum()),held_states=len(held_ids),accuracy=accuracy,greedy_accuracy=base,accuracy_gain=accuracy-base,
 mean_NLL=float(nll.mean()),control_mean_NLL=float(control_nll.mean()),delta_NLL=float((nll-control_nll).mean()),delta_NLL_day_CI95=ci,
 no_time_targets_in_features=True,Validation_access=False,Test_access=False,source_sha256=sha(run/'fuente_ejecutada.py'),
 predictions_sha256=sha(run/'predicciones_held_Train.npz'),model_sha256=sha(run/'politica.cbm') if passed else None,
 limitation='Teacherstate top1 accuracy compounds along whole generatedroutes; trueendpointroad is privileged here. A separate raw5publicendpoint ODgate is required before full routearchitecture.',
 seconds=time.time()-start)
write(run/'resultado.json',result);print(json.dumps({k:result[k] for k in ['passed','accuracy','greedy_accuracy','accuracy_gain','mean_NLL','control_mean_NLL','seconds']}),flush=True)
