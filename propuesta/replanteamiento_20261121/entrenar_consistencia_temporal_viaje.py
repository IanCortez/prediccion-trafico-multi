"""Two fixed full-Train epochs of aggregate journey-time correction; one GPU only."""
from base_replanteo import *
import torch,shutil,gc,math,argparse
from inferencia import feature_input
from datos_progreso_recurrente import CorrectedSequences,load_graph
from modelo_consistencia_temporal_viaje import ConsistentProgressNet
ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');args=ap.parse_args()
access=deny_test();start=time.time();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
torch.cuda.set_per_process_memory_fraction(.5);torch.manual_seed(20261122);torch.cuda.manual_seed_all(20261122)
previous=read(OUT/'progreso_vial_recurrente_s0/etapa_B/resultado.json')
assert previous['complete'],'Finish previous fullprimary evaluation first'
proof=read(OUT/'verificacion_consistencia_temporal_viaje.json')
assert proof['passed'] and proof['model_source_sha256']==sha(OUT/'modelo_consistencia_temporal_viaje.py')
assert proof['parent_source_sha256']==sha(OUT/'modelo_progreso_recurrente.py')
initfolder=OUT/'inicializacion_consistencia_temporal_s0';init=read(initfolder/'resultado.json')
assert sha(initfolder/'inicial.pt')==init['model_sha256']
run=OUT/'consistencia_temporal_viaje_s0';sources=['entrenar_consistencia_temporal_viaje.py','modelo_consistencia_temporal_viaje.py',
 'modelo_progreso_recurrente.py','datos_progreso_recurrente.py','base_replanteo.py']
config=dict(seed=20261122,epochs=2,parent_batch=256,fragment_max=128,lr=.0007,final_lr=.00007,weight_decay=.0003,clip=5,
 trained_parameters=8386,body='Frozenroot/GRU/roadembeddings/policy/valuehead; onlypositivepace/delaycorrection trained.',
 initialization=init,loss='MeanparentL1(sumknownpredictedlocaltimes,sumoriginalknownTrainlocaltimes)/600 +.001meanboundedcorrectionSquared.',
 target='Aggregateknown portion of originalTrainT, notloglocalHuber. Everyauxparentweightedonce; no inverseobservedN/durationweight.',
 style='PRIVATEteacher prefix/clock/positions but PUBLICprior-mode style; no recognitionposterior or targetT predictorinput.',
 original_primary_Train=1023485,primary_filter_changes=False,final_point_stage='AlloriginalTrainT andunchangedValidation.',
 query='Samepublicraw5; ownpredictedroute/clock. Top4pairs*4styles,192stepcap,30summaries unchanged.',
 stopping='Fixedtwoepochs; noValidationselection orlearningrate/count/stepcapseach.',
 reuse_control='Reusepreviousfull216CatBoostcontrol: identicaldataset/features/hyperparameters; no redundantrefit.',
 source_parent='progreso_vial_recurrente_s0/resultado_etapa_A.json',Test='sealed')
if not args.resume:
 assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
 for name in sources:shutil.copy2(OUT/name,run/'fuentes'/name)
 write(run/'protocolo.json',config)
else:
 assert read(run/'protocolo.json')==config
 for name in sources:assert sha(OUT/name)==sha(run/'fuentes'/name)
state=torch.load(initfolder/'inicial.pt',map_location='cpu',weights_only=False)
model=ConsistentProgressNet(state['state']['mean'],state['state']['std'],load_graph())
missing,unexpected=model.load_state_dict(state['state'],strict=False)
assert set(missing)=={'time_correction.0.weight','time_correction.0.bias','time_correction.2.weight','time_correction.2.bias'} and not unexpected
model=model.cuda();features=state['features'];del state
teacher=CorrectedSequences();d,X,names=feature_input('train',read(BEST/'resultado.json'));assert names==features
x=torch.tensor(X,device='cuda');del X;gc.collect()
f=teacher.fragments;sort=np.argsort(f[:,2],kind='stable')
counts=np.bincount(f[:,2],minlength=len(teacher.y));ptr=np.r_[0,np.cumsum(counts)];parents=np.flatnonzero(counts)
assert len(parents)==894104
with torch.no_grad():table=model.roads(torch.arange(model.nroads+1,device='cuda')).detach()
params=list(model.time_correction.parameters());assert sum(p.numel() for p in params)==8386
opt=torch.optim.AdamW(params,lr=config['lr'],weight_decay=config['weight_decay'])
def body_hash():
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):
  if not k.startswith('time_correction.'):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
initial_body=body_hash()
def batch(parent_ids):
 lens=counts[parent_ids];prefix=np.r_[0,np.cumsum(lens)]
 rank=np.repeat(ptr[parent_ids]-prefix[:-1],lens)+np.arange(prefix[-1])
 fi=sort[rank];slots=torch.as_tensor(np.repeat(np.arange(len(parent_ids)),lens),device='cuda')
 b=teacher.batch(fi)
 assert np.array_equal(b['train_index'].cpu().numpy(),np.repeat(parent_ids,lens))
 return b,slots
def objective(parent_ids):
 b,slots=batch(parent_ids);ii=b['train_index']
 local,correction=model.private_local_prediction(x[ii],b,table)
 pred=torch.zeros(len(parent_ids),device='cuda').index_add_(0,slots,(local*b['valid']).sum(1))
 truth=torch.zeros_like(pred).index_add_(0,slots,(b['local_s']*b['valid']).sum(1))
 penalty=(correction.square().sum(-1)*b['valid']).sum()/b['valid'].sum()/2
 return torch.abs(pred-truth).mean()/600+.001*penalty,pred,truth,len(local),int(b['valid'].sum())
sampled=np.load(OUT/'diagnostico_suma_tiempos_progreso_Train.npz')['train_index']
@torch.no_grad()
def matched_diagnostic():
 pred=[];true=[]
 for a in range(0,len(sampled),256):
  _,p,t,_,_=objective(sampled[a:a+256]);pred.append(p.cpu().numpy());true.append(t.cpu().numpy())
 return np.concatenate(pred),np.concatenate(true)
history=[];first=1;elapsed_before=0
if args.resume:
 ck=torch.load(run/'reanudacion.pt',map_location='cpu',weights_only=False)
 model.load_state_dict(ck['state']);opt.load_state_dict(ck['optimizer']);history=ck['history'];first=ck['epoch']+1;elapsed_before=ck['elapsed'];del ck
 initial=np.load(run/'diagnostico_antes_Train.npz');before=initial['pred'];known_true=initial['true'];initial.close()
else:
 before,known_true=matched_diagnostic()
 np.savez_compressed(run/'diagnostico_antes_Train.npz',train_index=sampled,pred=before,true=known_true)
 model.train();opt.zero_grad(set_to_none=True);pre,_,_,_,_=objective(parents[:256]);assert torch.isfinite(pre);pre.backward()
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
 assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('time_correction.'))
 opt.zero_grad(set_to_none=True)
 write(run/'verificacion_GPU.json',dict(passed=True,real256parentgradient_finite=True,
  original_frozen_body_unchanged=body_hash()==initial_body,GPU_peak_MiB=torch.cuda.max_memory_allocated()/1024**2,
  state_MiB=sum(v.numel()*v.element_size() for v in model.state_dict().values())/1024**2,
  known_teacher_tokens=41511860,originalprimary_queries=1023485,Test=False))
for epoch in range(first,3):
 model.train();order=np.random.default_rng(8700+epoch).permutation(parents);tot=0.;n=0;tokens=0;fragments=0;tick=time.time()
 steps=(len(order)+255)//256
 for step,a in enumerate(range(0,len(order),256)):
  progress=((epoch-1)+step/steps)/2;lr=config['final_lr']+(config['lr']-config['final_lr'])*(.5+.5*math.cos(math.pi*progress))
  for group in opt.param_groups:group['lr']=lr*min(1.,((epoch-1)*steps+step+1)/100)
  ids=order[a:a+256];opt.zero_grad(set_to_none=True)
  loss,p,t,nf,nt=objective(ids);assert torch.isfinite(loss);loss.backward()
  torch.nn.utils.clip_grad_norm_(params,5,error_if_nonfinite=True);opt.step()
  tot+=float(torch.abs(p-t).sum().detach());n+=len(ids);tokens+=nt;fragments+=nf
  if step%200==0:
   report=dict(epoch=epoch,step=step,steps=steps,parent_n=n,online_training_knownT_MAE_s=tot/n,seconds=time.time()-tick)
   write(run/'progreso.json',report);print(json.dumps(report),flush=True)
 assert n==894104 and fragments==2099324 and tokens==41511860 and body_hash()==initial_body
 row=dict(epoch=epoch,parents=n,fragments=fragments,known_tokens=tokens,online_training_knownT_MAE_s=tot/n,seconds=time.time()-tick)
 history.append(row);write(run/'historia.json',history)
 torch.save(dict(state=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,history=history,elapsed=elapsed_before+time.time()-start),run/'reanudacion.pt')
model.eval();after,again=matched_diagnostic();assert np.array_equal(again,known_true)
np.savez_compressed(run/'diagnostico_despues_Train.npz',train_index=sampled,true=known_true,pred=after,parent_pred=before)
model_file=run/'modelo_consistencia_epoca2.pt'
torch.save(dict(state=model.state_dict(),config=config,features=features,epoch=2),model_file)
assert model_file.stat().st_size<50*1024**2 and body_hash()==initial_body
result=dict(complete=True,stage='Correctiontrained; completepublicODrollouts andpointregression stillpending',
 model_file=str(model_file),model_sha256=sha(model_file),features=features,source_parent_sha256=init['model_sha256'],
 frozen_body_unchanged=True,body_sha256=initial_body,private_Train_diagnostic_before=metric(known_true,before),
 private_Train_diagnostic_after=metric(known_true,after),diagnostic_scope='512Trainknownpath/clock; notODValidation.',
 sources={n:sha(run/'fuentes'/n) for n in sources},Test_access=False,actual_data_files_accessed=sorted(access),
 seconds=elapsed_before+time.time()-start)
write(run/'resultado_modelo.json',result)
p=run/'reanudacion.pt';assert p.resolve().is_relative_to(run.resolve());p.unlink()
print(json.dumps(result),flush=True)
