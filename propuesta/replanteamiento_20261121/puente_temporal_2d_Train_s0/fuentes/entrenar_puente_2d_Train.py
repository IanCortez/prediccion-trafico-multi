"""Threefreshmatched allprecutoffTrain arms, fixed4fullpasses; held evaluated only at finalstate."""
from base_replanteo import *
import torch,math,gc,shutil
from datos_puente_temporal_2d import BridgeInputs
from modelo_puente_temporal_2d import TemporalBridge2D,canonical_predict
from puente_temporal_2d import duration_logp,score,conditional_fit_logpdf,quantiles
start=time.time();access=deny_test()
run=OUT/'puente_temporal_2d_Train_s0';assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
config=read(OUT/'protocolo_puente_2d_Train.json');assert config['status']=='FROZEN_BEFORE_ANY_NEW_HELDRESULT'
preflight=read(OUT/'preflight_puente_2d_GPU_s0/resultado.json');assert preflight['passed']
for name,digest in preflight['sources'].items():assert sha(OUT/name)==digest
sources=['entrenar_puente_2d_Train.py','datos_puente_temporal_2d.py','modelo_puente_temporal_2d.py',
 'puente_temporal_2d.py','coordenadas_trayecto_2d.py','modelo_mezcla_individual_mediana.py','base_replanteo.py']
for name in sources:shutil.copy2(OUT/name,run/'fuentes'/name)
write(run/'protocolo.json',config);write(run/'diseno_refinado.json',read(OUT/'refinamiento_puente_temporal_2D.json'))
write(run/'fuentes_publicas.json',{str(p.relative_to(FROZEN)):sha(p) for p in FROZEN.glob('*.py')})
torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(.5)
data=BridgeInputs(run);steps=math.ceil(data.nfit/256);assert steps==3718 and steps*4==14872
def make():
 torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
 return TemporalBridge2D(**data.constructor).cuda()
def parameter_sha(model):
 h=hashlib.sha256()
 for key,p in model.named_parameters():h.update(key.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
metrics={};probability_scores={};allq={};checkpoints=[]
for mode in config['arms']:
 model=make();initial=parameter_sha(model);assert initial==preflight['initial_parameters_sha256']
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0003)
 schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=steps*4,eta_min=.00001)
 arm=run/mode;arm.mkdir();history=[]
 for epoch in range(1,5):
  torch.manual_seed(config['seed']+epoch);torch.cuda.manual_seed_all(config['seed']+epoch)
  order=torch.randperm(data.nfit,device='cuda',generator=torch.Generator(device='cuda').manual_seed(config['seed']+1000+epoch))
  model.train();totals=torch.zeros(5,device='cuda',dtype=torch.float64);count=0
  for step,ids in enumerate(order.split(256),1):
   opt.zero_grad(set_to_none=True);params=model(data.x[ids]);logp,end=duration_logp(params)
   n=data.target_n[ids];risk,crps,nll=score(logp,n)
   if mode=='none':aux=torch.zeros_like(risk)
   else:
    increments,nt=data.fit_targets(ids,mode);assert torch.equal(n,nt)
    aux=-conditional_fit_logpdf(increments,n,params,end)/(2*(n-1))
   loss=(risk+.02*aux).mean();assert torch.isfinite(loss)
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step();schedule.step()
   point=quantiles(logp.detach())[:,1];mae=(point-n*15).abs().double().mean()
   size=len(ids);totals+=torch.stack([loss.detach(),crps.detach().mean(),nll.detach().mean(),aux.detach().mean(),mae])*size;count+=size
   if step%512==0:print(json.dumps(dict(stage='bridge_updates',arm=mode,epoch=epoch,step=step,total_steps=steps,seen=count,seconds=time.time()-start)),flush=True)
  assert count==data.nfit;values=totals.cpu().numpy()/count
  row=dict(arm=mode,epoch=epoch,train_n=count,loss=float(values[0]),train_CRPS_s=float(values[1]),train_binNLL=float(values[2]),
   train_auxNLL_perinteriordimension=float(values[3]),train_MAE_s=float(values[4]),updates=steps,lr=schedule.get_last_lr()[0],seconds=time.time()-start)
  history.append(row);write(arm/'historia.json',history);print(json.dumps(row),flush=True)
 file=arm/'epoca_4.pt'
 torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},constructor=data.constructor,
  config=config,epoch=4,features=data.names),file)
 logp,q=canonical_predict(model,data.x[data.nfit:]);lp=logp.numpy();cq=q.numpy()
 truth=data.truth[data.nfit:];nt=torch.tensor((truth/15).astype(np.int64));_,crps,nll=score(logp,nt)
 metrics[mode]=metric(truth,cq[:,1]);allq[mode]=cq
 probability_scores[mode]=dict(CRPS_s=float(crps.mean()),bin_NLL=float(nll.mean()),
  coverage80=float(np.mean((truth>=cq[:,0])&(truth<=cq[:,2]))),width80_s=float((cq[:,2]-cq[:,0]).mean()))
 predictions=arm/'predicciones_held_Train.npz'
 np.savez_compressed(predictions,row=data.row[data.nfit:],timestamp=data.timestamp[data.nfit:],true=truth,pred=cq[:,1],quantiles=cq,log_prob=lp.astype(np.float32))
 item=dict(arm=mode,file=str(file),sha256=sha(file),metrics=metrics[mode],probability_scores=probability_scores[mode],
  predictions_sha256=sha(predictions),scope='All10000heldTrain only, notprimaryValidation')
 checkpoints.append(item)
 write(arm/'resultado.json',dict(complete=True,fixed_epoch=4,initial_parameters_sha256=initial,**item))
 print(json.dumps(dict(stage='bridge_arm_complete',arm=mode,metrics=metrics[mode],probability_scores=probability_scores[mode],seconds=time.time()-start)),flush=True)
 del model,opt,schedule,logp,end,params,risk,loss,aux,order;gc.collect();torch.cuda.empty_cache()
reference=dict(np.load(OUT/'oraculo_tiempo_calle_train_s0/predicciones_held_Train.npz'))
for key,value in [('row',data.row[data.nfit:]),('timestamp',data.timestamp[data.nfit:]),('true',data.truth[data.nfit:])]:assert np.array_equal(reference[key],value)
truth=data.truth[data.nfit:];times=data.timestamp[data.nfit:]
comparisons={key:paired(truth,allq['observed'][:,1],value,times) for key,value in
 [('vs_neuralbaseline',allq['none'][:,1]),('vs_linearcontrol',allq['linear'][:,1]),('vs_CatBoostreference',reference['control'])]}
for value in comparisons.values():value['interpretation']='ExploratoryreusedheldTrain; oneinitialization, notprimaryValidation or confirmation.'
passed=all(comparisons[key]['delta_mae_s']<=-threshold and comparisons[key]['ci95_s'][1]<0 for key,threshold in
 [('vs_neuralbaseline',5),('vs_linearcontrol',2),('vs_CatBoostreference',5)])
np.savez_compressed(run/'predicciones_held_Train.npz',row=data.row[data.nfit:],timestamp=times,true=truth,reference=reference['control'],
 **{key:value[:,1] for key,value in allq.items()},**{key+'_quantiles':value for key,value in allq.items()})
result=dict(complete=True,passed=bool(passed),fit_n=data.nfit,held_n=10000,fixed_epoch=4,updates_per_arm=steps*4,
 metrics=metrics,probability_scores=probability_scores,comparisons=comparisons,reference=metric(truth,reference['control']),
 initial_parameters_sha256=initial,all_matched_fresh_initializations=True,checkpoints=checkpoints,independent_replay_pending=True,
 predictions_sha256=sha(run/'predicciones_held_Train.npz'),constructor_sha256=sha(run/'constructor.pt'),
 sources={name:sha(run/'fuentes'/name) for name in sources},protocol_sha256=sha(run/'protocolo.json'),
 no_query_trueN_or_GPS_in_prediction=True,original_primary_cohort_queries_labels_unchanged=True,
 source_stream='FITsupervision only; no newpopulationcache, no heldGPS used.',
 Test_access=False,Validation_access=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start,
 next='Independentall10000/raw65replay before retention/promotion. Samegates; no additionalepochs or parametergrid.')
write(run/'resultado.json',result);print(json.dumps(result),flush=True)
