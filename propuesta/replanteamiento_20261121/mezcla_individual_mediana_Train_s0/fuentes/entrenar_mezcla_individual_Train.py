"""Fresh fixed80epoch Train-only individual-source mixture and paired controls."""
from base_replanteo import *
import torch,gc,shutil,math
from datos_mezcla_individual_Train import prepare_inputs
from modelo_mezcla_individual_mediana import IndividualSourceMixture,canonical_quantiles
from cuantiles_mezcla_implicita import mixture_loss
access=deny_test();start=time.time()
proof=read(OUT/'verificacion_mediana_control_arco_v3/resultado.json');assert proof['passed']
for name,digest in proof['sources'].items():assert sha(OUT/name)==digest
run=OUT/'mezcla_individual_mediana_Train_s0';assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
sources=['entrenar_mezcla_individual_Train.py','datos_mezcla_individual_Train.py','modelo_mezcla_individual_mediana.py','cuantiles_mezcla_implicita.py',
 'tokens_transferencia_arco.py','tokens_subtrayectos_arco.py','tokens_subtrayectos.py','recuperar_fuentes_ordenadas.py','recuperar_subtrayectos.py','base_replanteo.py']
for name in sources:shutil.copy2(OUT/name,run/'fuentes'/name)
config=dict(seed=20261125,epochs=80,batch=256,lr=.001,min_lr=.00001,weight_decay=.0003,clip=5,members=4,global_components=8,source_components=64,source_features=38,
 query=['o_lon','o_lat','d_lon','d_lat','timestamp'],fit_n=40000,held_n=10000,epoch_selection='Fixedfinal80, no heldstopping or heldmetrics duringtraining.',
 objective='Actualensemble-mixturemedian MAE/600 +.02 ensembled original15secondbinNLL; foldedtails60..1785.',
 initial='Freshmatchedrandomneuralparameters, no parent pretrained onheldlabels.',arms=['none','geometry','observed'],
 practical_gate='>=5s versus freshneuralbaseline andstoredCatBoost133.355393; >=2s versus equalcapacitygeometryproxy, all dayCIupper<0.',
 retention='Allfinalweights transient until independentreplay; retainpilotweights onlyif gatepassed. Everycompleted prediction/metric/config/source alwaysretained.',
 Test='SEALED',Validation='Not accessed',resources='OneGPU .5cap; no persistent rawtokens or extraGPSbank.',canonical_batch=256)
write(run/'protocolo.json',config);write(run/'plan_diseno.json',read(OUT/'plan_mezcla_individual_mediana_Train.json'))
torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.use_deterministic_algorithms(False);torch.cuda.set_per_process_memory_fraction(.5)
data=prepare_inputs(run);x=data['X'];y=torch.tensor(data['y'][:40000],device='cuda')
def make(mode):
 torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
 normal=data['norm']['geometry' if mode=='geometry' else 'observed']
 return IndividualSourceMixture(**data['constructor'],rmean=normal['mean'],rstd=normal['std'],source_mode=mode).cuda()
def param_sha(model):
 h=hashlib.sha256()
 for name,p in model.named_parameters():h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
# Actual resident-data two-step gradient/memory gate; this temporarymodel is discarded.
m=make('observed');initial=param_sha(m);torch.cuda.reset_peak_memory_stats();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0003)
for step in range(2):
 m.train();opt.zero_grad(set_to_none=True);loss,mae,nll=mixture_loss(m(x[:256],data['observed'][:256]),y[:256]);assert torch.isfinite(loss)
 loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True)
 assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None);opt.step()
assert any(p.grad is not None and float(p.grad.abs().sum())>0 for p in m.members[0].query.parameters())
check=canonical_quantiles(m,x[:17],data['observed'][:17]);rev=canonical_quantiles(m,x[:17].flip(0),data['observed'][:17].flip(0))
single=canonical_quantiles(m,x[:1],data['observed'][:1]);assert torch.equal(check,rev.flip(0)) and torch.equal(check[:1],single)
write(run/'preflight_GPU.json',dict(passed=True,resident_all50000_inputs=True,train_batch=256,finite_two_step_gradients=True,query_backbone_gradient_nonzero=True,
 inference_canonical256_order_singleton_exact=True,peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,initial_neural_sha256=initial))
print(json.dumps(dict(stage='GPU_preflight_passed',peak_MiB=torch.cuda.max_memory_allocated()/1024**2,seconds=time.time()-start)),flush=True)
del opt,m;gc.collect();torch.cuda.empty_cache()
all_predictions={};metrics={};checkpoints=[];history=[]
for mode in config['arms']:
 m=make(mode);assert param_sha(m)==initial
 raw=data['geometry' if mode=='geometry' else 'observed']
 opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0003);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=80,eta_min=.00001)
 arm=run/mode;arm.mkdir();arm_history=[]
 for epoch in range(1,81):
  torch.manual_seed(config['seed']+epoch);torch.cuda.manual_seed_all(config['seed']+epoch)
  order=torch.randperm(40000,device='cuda',generator=torch.Generator(device='cuda').manual_seed(config['seed']+1000+epoch))
  m.train();total=0.;total_mae=0.;total_nll=0.;count=0
  for ids in order.split(256):
   opt.zero_grad(set_to_none=True);loss,mae,nll=mixture_loss(m(x[ids],raw[ids]),y[ids]);assert torch.isfinite(loss)
   loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step()
   n=len(ids);total+=float(loss.detach())*n;total_mae+=float(mae.detach())*600*n;total_nll+=float(nll.detach())*n;count+=n
  assert count==40000;schedule.step()
  item=dict(arm=mode,epoch=epoch,train_n=count,loss=total/count,train_mae_s=total_mae/count,bin_NLL=total_nll/count,lr=schedule.get_last_lr()[0],seconds=time.time()-start)
  arm_history.append(item);history.append(item);write(arm/'historia.json',arm_history)
  if epoch%5==0 or epoch==1:print(json.dumps(item),flush=True)
 # Freeze the fixedfinal state before seeing any heldresult.
 path=arm/'epoca_80.pt';normal=data['norm']['geometry' if mode=='geometry' else 'observed']
 ctor=dict(data['constructor'],rmean=normal['mean'],rstd=normal['std'],source_mode=mode)
 torch.save(dict(state={k:v.detach().cpu() for k,v in m.state_dict().items()},constructor=ctor,config=config,epoch=80,features=data['names']),path)
 cq=canonical_quantiles(m,x[40000:],raw[40000:]).numpy();assert np.isfinite(cq).all()
 all_predictions[mode]=cq;metrics[mode]=metric(data['y'][40000:],cq[:,1])
 checkpoints.append(dict(arm=mode,file=str(path),sha256=sha(path),validation_scope='HeldTrain only, not primaryValidation',metrics=metrics[mode]))
 write(arm/'resultado.json',dict(complete=True,fixed_epoch=80,metrics=metrics[mode],model_file=str(path),model_sha256=sha(path),initial_neural_sha256=initial,
 source_mode=mode,held_prediction_sha256=None,Test=False,Validation=False))
 # Every completed arm is immediately saved independently, even if a laterarmfails.
 np.savez_compressed(arm/'predicciones_held_Train.npz',row=data['q']['row'][40000:],timestamp=data['q']['timestamp'][40000:],true=data['y'][40000:],pred=cq[:,1],quantiles=cq)
 print(json.dumps(dict(stage='arm_complete',arm=mode,metrics=metrics[mode],seconds=time.time()-start)),flush=True)
 del opt,schedule,m;gc.collect();torch.cuda.empty_cache()
reference=dict(np.load(OUT/'oraculo_tiempo_calle_train_s0/predicciones_held_Train.npz'))
for key in ['row','timestamp']:assert np.array_equal(reference[key],data['q'][key][40000:])
assert np.array_equal(reference['true'],data['y'][40000:])
truth=data['y'][40000:];times=data['q']['timestamp'][40000:]
comparisons={key:paired(truth,all_predictions['observed'][:,1],other,times) for key,other in
 [('vs_neuralbaseline',all_predictions['none'][:,1]),('vs_geometryproxy',all_predictions['geometry'][:,1]),('vs_CatBoostreference',reference['control'])]}
for item in comparisons.values():item['interpretation']='Exploratory reusedchronologicalheldTrain, not primaryValidation or confirmation.'
passed=all(comparisons[key]['delta_mae_s']<=-delta and comparisons[key]['ci95_s'][1]<0 for key,delta in
 [('vs_neuralbaseline',5),('vs_geometryproxy',2),('vs_CatBoostreference',5)])
np.savez_compressed(run/'predicciones_held_Train.npz',row=data['q']['row'][40000:],timestamp=times,true=truth,reference=reference['control'],
 **{key:value[:,1] for key,value in all_predictions.items()},**{key+'_quantiles':value for key,value in all_predictions.items()})
result=dict(complete=True,passed=bool(passed),metrics=metrics,comparisons=comparisons,reference=metric(truth,reference['control']),fixed_epoch=80,
 all_neural_initial_parameters_matched=True,initial_neural_sha256=initial,checkpoints=checkpoints,independent_replay_pending=True,
 predictions_sha256=sha(run/'predicciones_held_Train.npz'),constructor_sha256=sha(run/'constructor.pt'),
 sources={name:sha(run/'fuentes'/name) for name in sources},Validation_access=False,Test_access=False,
 actual_data_files_accessed=sorted(access),seconds=time.time()-start,
 next='Independentfullheldreload plus raw65query replay required before retentiondecision; do not change epochs/gates after results.')
write(run/'resultado.json',result);print(json.dumps(read(run/'resultado.json')),flush=True)
