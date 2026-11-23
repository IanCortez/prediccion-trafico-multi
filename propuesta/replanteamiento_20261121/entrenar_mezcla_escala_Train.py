"""Fixed4fullpass comparison with allprecutoffTrain, streamedtokens and stableloss."""
from base_replanteo import *
import torch,gc,shutil,math
from datos_mezcla_escala_Train import StreamInputs
from modelo_mezcla_individual_mediana import IndividualSourceMixture
from cuantiles_mezcla_balanceada import stable_mixture_loss
access=deny_test();start=time.time()
source=OUT/'indices_escala_Train_s0';s=read(source/'resultado.json');assert s['complete']
plan=read(OUT/'plan_escala_datos_Train.json');assert s['fit_n']==plan['source_fit_n']
assert read(OUT/'verificacion_cuantiles_balanceados_s0/resultado.json')['passed'] and read(OUT/'preflight_balanceado_GPU.json')['passed']
run=OUT/'mezcla_individual_escala_Train_s0';assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
sources=['entrenar_mezcla_escala_Train.py','datos_mezcla_escala_Train.py','modelo_mezcla_individual_mediana.py','cuantiles_mezcla_balanceada.py',
 'tokens_transferencia_arco.py','tokens_subtrayectos_arco.py','tokens_subtrayectos.py','datos_mezcla_individual_Train.py',
 'recuperar_fuentes_ordenadas.py','recuperar_subtrayectos.py','datos_progreso_recurrente.py','base_replanteo.py']
for name in sources:shutil.copy2(OUT/name,run/'fuentes'/name)
config=dict(seed=20261125,fit_n=s['fit_n'],held_n=10000,epochs=4,batch=256,members=4,global_components=8,source_components=64,source_features=38,
 lr=.001,min_lr=.00001,schedule='Per-update cosine across exactly14872updates; no heldstopping.',
 weight_decay=.0003,clip=5,arms=['none','geometry','observed'],canonical_batch=256,
 source_normalization_queries=32768,source_normalization_seed=20261127,objective='Stable exactensemblemedian MAE/600+.02 stableoriginal15secondbinNLL; onlyextremeweightgradientbounded1e6s.',
 query=['o_lon','o_lat','d_lon','d_lat','timestamp'],precision='FP32 neuralweights/highest/noTF32; Float64 signedlogtail quantiles/probabilities.',
 epoch_selection='Fixedfinal4, all951718fitrows once each epoch. No heldmetric beforefinalstate.',
 practical_gate='>=5s betterthan matchedfreshneuralbaseline and storedCatBoost133.355393;>=2s betterthan geometryproxy, all exploratorydayCIupper<0.',
 initial='Freshmatchedneuralparameters, same seedas smallpilot; no pretrainedheldlabels.',
 resources='OneGPU .5cap; resident publicX/sourceindices/pointbank only; rawtokens perB256, no population rawtokencache.',
 preservation='Every completed metric/prediction/config/source retained; finalpilotweights transient until independentreplay and gate decision.',Test='SEALED',Validation=False)
write(run/'protocolo.json',config);write(run/'plan_diseno.json',plan)
torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.use_deterministic_algorithms(False);torch.cuda.set_per_process_memory_fraction(.5)
data=StreamInputs(run);steps=math.ceil(data.nfit/256);assert steps==3718 and steps*4==14872
def make(mode):
 torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
 norm=data.norm['geometry' if mode=='geometry' else 'observed']
 return IndividualSourceMixture(**data.common,rmean=norm['mean'],rstd=norm['std'],source_mode=mode).cuda()
def param_sha(model):
 h=hashlib.sha256()
 for name,p in model.named_parameters():h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
m=make('observed');initial=param_sha(m);torch.cuda.reset_peak_memory_stats()
opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0003);probe=torch.arange(256,device='cuda')
for k in range(2):
 m.train();opt.zero_grad(set_to_none=True);x,r=data.batch(probe,'observed')
 loss,mae,nll=stable_mixture_loss(m(x,r),data.y[probe]);assert torch.isfinite(loss);loss.backward()
 torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None);opt.step()
assert any(p.grad is not None and float(p.grad.abs().sum())>0 for p in m.members[0].query.parameters())
# Heldfeatures only: rawfivefield publicbuilder/tokenstream equality, without heldtargets or modelupdates.
from datos_mezcla_individual_Train import build_public206,tensor_tokens
probe_ids=np.linspace(data.nfit,data.n-1,65,dtype=int)
items=[{key:int(value[i]) if key=='timestamp' else float(value[i]) for key,value in data.queries.items()} for i in probe_ids]
rawX,rawnames=build_public206(items,read(BEST/'resultado.json')['config'])
assert rawnames==data.names and np.array_equal(rawX,data.x[torch.tensor(probe_ids,device='cuda')].cpu().numpy())
query={key:value[probe_ids] for key,value in data.queries.items()}
rr=dict(parent=data.parent[probe_ids].cpu().numpy(),first=data.first[probe_ids].cpu().numpy(),last=data.last[probe_ids].cpu().numpy())
for mode in ['observed','geometry']:
 direct=tensor_tokens(data.bank,query,rawX,**rr,shape_indices=data.shape_indices,available_indices=data.available_indices,geometry_proxy=mode=='geometry')
 _,streamed=data.batch(probe_ids,mode,canonical=True)
 assert torch.equal(direct,streamed[:65])
write(run/'verificacion_stream_crudo.json',dict(passed=True,raw65_public206_exact=True,raw65_streamed38Dtokens_exact=True,heldlabels_not_used=True))
p=data.predict(m,'observed',np.arange(17));rev=data.predict(m,'observed',np.arange(17)[::-1]);single=data.predict(m,'observed',np.array([0]))
assert np.array_equal(p,rev[::-1]) and np.array_equal(p[:1],single)
write(run/'preflight_GPU.json',dict(passed=True,fit_n=data.nfit,resident_all_publicX_and_sourceindices=True,no_full_rawtokens=True,
 peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,two_step_gradients_finite=True,query_backbone_nonzero=True,
 canonical256_reverse_singleton_exact=True,initial_neural_sha256=initial))
print(json.dumps(dict(stage='actual_all_data_GPU_preflight_passed',peak_MiB=torch.cuda.max_memory_allocated()/1024**2,seconds=time.time()-start)),flush=True)
del m,opt,x,r;gc.collect();torch.cuda.empty_cache()
allq={};metrics={};checkpoints=[]
for mode in config['arms']:
 model=make(mode);assert param_sha(model)==initial
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0003)
 schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=steps*4,eta_min=.00001)
 arm=run/mode;arm.mkdir();history=[]
 for epoch in range(1,5):
  torch.manual_seed(config['seed']+epoch);torch.cuda.manual_seed_all(config['seed']+epoch)
  order=torch.randperm(data.nfit,device='cuda',generator=torch.Generator(device='cuda').manual_seed(config['seed']+1000+epoch))
  model.train();totals=torch.zeros(3,device='cuda',dtype=torch.float64);count=0
  for step,ids in enumerate(order.split(256),1):
   x,r=data.batch(ids,mode);opt.zero_grad(set_to_none=True)
   loss,mae,nll=stable_mixture_loss(model(x,r),data.y[ids]);assert torch.isfinite(loss)
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step();schedule.step()
   n=len(ids);totals+=torch.stack([loss.detach(),mae.detach()*600,nll.detach()])*n;count+=n
   if step%512==0:print(json.dumps(dict(stage='all_data_updates',arm=mode,epoch=epoch,step=step,total_steps=steps,seen=count,seconds=time.time()-start)),flush=True)
  assert count==data.nfit;value=totals.cpu().numpy()/count
  row=dict(arm=mode,epoch=epoch,train_n=count,loss=float(value[0]),train_mae_s=float(value[1]),bin_NLL=float(value[2]),lr=schedule.get_last_lr()[0],updates=steps,seconds=time.time()-start)
  history.append(row);write(arm/'historia.json',history);print(json.dumps(row),flush=True)
 path=arm/'epoca_4.pt';norm=data.norm['geometry' if mode=='geometry' else 'observed']
 ctor=dict(data.common,rmean=norm['mean'],rstd=norm['std'],source_mode=mode)
 torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},constructor=ctor,config=config,epoch=4,features=data.names),path)
 cq=data.predict(model,mode);allq[mode]=cq;metrics[mode]=metric(data.truth[data.nfit:],cq[:,1])
 file=arm/'predicciones_held_Train.npz'
 np.savez_compressed(file,row=data.row[data.nfit:],timestamp=data.timestamp[data.nfit:],true=data.truth[data.nfit:],pred=cq[:,1],quantiles=cq)
 item=dict(arm=mode,file=str(path),sha256=sha(path),metrics=metrics[mode],scope='HeldTrain only, not primaryValidation',predictions_sha256=sha(file))
 checkpoints.append(item);write(arm/'resultado.json',dict(complete=True,fixed_epoch=4,initial_neural_sha256=initial,**item))
 print(json.dumps(dict(stage='all_data_arm_complete',arm=mode,metrics=metrics[mode],seconds=time.time()-start)),flush=True)
 del model,opt,schedule,x,r;gc.collect();torch.cuda.empty_cache()
reference=dict(np.load(OUT/'oraculo_tiempo_calle_train_s0/predicciones_held_Train.npz'))
for k,v in [('row',data.row[data.nfit:]),('timestamp',data.timestamp[data.nfit:]),('true',data.truth[data.nfit:])]:assert np.array_equal(reference[k],v)
truth=data.truth[data.nfit:];times=data.timestamp[data.nfit:]
comparisons={k:paired(truth,allq['observed'][:,1],v,times) for k,v in
 [('vs_neuralbaseline',allq['none'][:,1]),('vs_geometryproxy',allq['geometry'][:,1]),('vs_CatBoostreference',reference['control'])]}
for v in comparisons.values():v['interpretation']='Exploratory reusedchronologicalheldTrain; not primaryValidation or confirmation.'
passed=all(comparisons[k]['delta_mae_s']<=-d and comparisons[k]['ci95_s'][1]<0 for k,d in
 [('vs_neuralbaseline',5),('vs_geometryproxy',2),('vs_CatBoostreference',5)])
np.savez_compressed(run/'predicciones_held_Train.npz',row=data.row[data.nfit:],timestamp=times,true=truth,reference=reference['control'],
 **{k:v[:,1] for k,v in allq.items()},**{k+'_quantiles':v for k,v in allq.items()})
result=dict(complete=True,passed=bool(passed),fit_n=data.nfit,held_n=10000,fixed_epoch=4,updates_per_arm=steps*4,
 metrics=metrics,comparisons=comparisons,reference=metric(truth,reference['control']),initial_neural_sha256=initial,matched_initial_parameters=True,
 checkpoints=checkpoints,independent_replay_pending=True,predictions_sha256=sha(run/'predicciones_held_Train.npz'),constructor_sha256=sha(run/'constructor.pt'),
 sourceindices_sha256=s['indices_sha256'],sources={name:sha(run/'fuentes'/name) for name in sources},
 original_queries_labels_and_targetcohort_unchanged=True,no_queryroute_N_T=True,Validation_access=False,Test_access=False,
 actual_data_files_accessed=sorted(access),seconds=time.time()-start,
 next='Independent fullheld/raw5 replay before retention/promotion. No extraepochs or threshold changes after outcome.')
write(run/'resultado.json',result);print(json.dumps(read(run/'resultado.json')),flush=True)
