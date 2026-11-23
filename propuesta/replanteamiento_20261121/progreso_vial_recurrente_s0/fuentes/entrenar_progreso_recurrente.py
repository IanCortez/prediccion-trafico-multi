"""Fixed six-epoch full-primary training; no Validation selection in sequence stage."""
from base_replanteo import *
import torch,shutil,gc,argparse,math
from inferencia import feature_input
from modelo_aux_subviajes import duration_score
from modelo_progreso_recurrente import ProgressNet
from datos_progreso_recurrente import CorrectedSequences,load_graph
ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');args=ap.parse_args()
access=deny_test();start=time.time();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
torch.cuda.set_per_process_memory_fraction(.5);torch.manual_seed(20261122);torch.cuda.manual_seed_all(20261122)
proof=read(OUT/'verificacion_progreso_recurrente.json')
assert proof['passed'] and proof['model_source_sha256']==sha(OUT/'modelo_progreso_recurrente.py')
assert proof['data_source_sha256']==sha(OUT/'datos_progreso_recurrente.py')
run=OUT/'progreso_vial_recurrente_s0'
sources=['entrenar_progreso_recurrente.py','modelo_progreso_recurrente.py','datos_progreso_recurrente.py',
 'modelo_aux_subviajes.py','base_replanteo.py']
config=dict(seed=20261122,epochs=6,auxiliary_batch=512,max_fragment=128,root_width=256,GRU=128,styles=4,
 public_top_pairs=4,public_max_steps=192,public_summary_width=30,
 learning_rate=.0007,final_learning_rate=.00007,weight_decay=.0003,clip=5,precision='FP32highest',
 loss='PrimaryNLL+5RPS + .2next/EOSCE+.2logLocalHuber+.1logRemainingHuber+.02styleKL+.1attachmentCE',
 primary_Train=1023485,auxiliary_Train=894105,auxiliary_tokens=41775036,
 source_teacher='Corrected full Train fragments, split ONLY at privatephysicaluncertainty or fixed128TBPTTboundary. Everyknownteacher token appears once/epoch; everyeligibleprimaryquery once/epoch.',
 style='FourTrainposteriorstyles; one parent-specific Gumbel draw per epoch reused acrossfragments. QueryusesrawODprior only.',
 clock='F64Unix input plus observedTrainclock / predictedqueryclock; publicLisbonDSTtable.',
 Validation='No selection/tuning in sixepochsequence training; rootquantiles evaluated only once at final.',
 stage_B='Freezeepoch6 then public-onlyrollouts for allTrain/Validation. Fitmatched1000tree depth8 GPUCatBoostMAE withandwithout30summaries plus threefullquantiles, fixedseed/depth/epoch, compareMAEpointmodel once.',
 gate='At least0.5seconds fullValidationgain over130.158 and upperdaybootstrapCI<0 AND >=.5secondgain overmatched216featureCatBoostcontrol before matchedprefix-resetcontrol; independentconfirmation73.5stillrequired.',
 preservation='Allcompletedmetrics/predictions, allcheckpointsbelow131.06 orstrictglobalbest; fixedepoch6dependency retained untilStageB evaluated. Dnonpersistent, modelstate<50MiB.',
 Test='sealed',reference=read(OUT/'nuevo_mejor_verificado.json'))
if not args.resume:
 assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
 for name in sources:shutil.copy2(OUT/name,run/'fuentes'/name)
 write(run/'protocolo.json',config)
else:
 assert read(run/'protocolo.json')==config
 for name in sources:assert sha(OUT/name)==sha(run/'fuentes'/name)
teacher=CorrectedSequences(config['max_fragment']);d,X,names=feature_input('train',read(BEST/'resultado.json'))
eligible=teacher.eligible
assert len(names)==216 and np.max(d['timestamp'][eligible]+teacher.y[eligible])<1398345209
mean=X[eligible].mean(0,dtype=np.float64).astype(np.float32);std=X[eligible].std(0,dtype=np.float64).clip(.01).astype(np.float32)
model=ProgressNet(mean,std,load_graph()).cuda();x=torch.tensor(X,device='cuda');del X;gc.collect()
np.savez_compressed(run/'normalizacion_Train.npz',mean=mean,std=std)
target=torch.as_tensor(teacher.y,device='cuda')
bankdata=dict(np.load(OUT/'banco_valor_vial_s0/train_proyecciones.npz'))
bank={k:torch.as_tensor(bankdata[k].astype(np.int64) if k=='chain' else bankdata[k],device='cuda') for k in ['chain','fraction','error_m','remaining_km','reachable']}
del bankdata
attachment=torch.as_tensor(teacher.index['attachment_pair'].astype(np.int64),device='cuda')
opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
history=[];first=1;elapsed_before=0
if args.resume:
 state=torch.load(run/'reanudacion.pt',map_location='cpu',weights_only=False)
 model.load_state_dict(state['state']);opt.load_state_dict(state['optimizer'])
 history=state['history'];first=state['epoch']+1;elapsed_before=state['elapsed']
 del state
else:
 write(run/'datos_verificados.json',dict(primary_n=len(eligible),teacher_fragments=len(teacher.fragments),
  teacher_tokens=len(teacher.code),teacher_RAM_MiB=teacher.bytes/1024**2,source_adapter_sha256=sha(OUT/'adaptador_progreso_vial_s0/resultado.json'),
  no_primary_filtered=True,all_source_arrivals_beforeValidation=True,Test=False))

def batch_indices(epoch):
 rng=np.random.default_rng(3000+epoch);batches=[]
 length=teacher.fragments[:,1]
 for lo,hi in [(0,16),(16,32),(32,64),(64,128)]:
  ids=np.flatnonzero((length>lo)&(length<=hi));rng.shuffle(ids)
  batches.extend([ids[a:a+config['auxiliary_batch']] for a in range(0,len(ids),config['auxiliary_batch'])])
 rng.shuffle(batches);primary=eligible.copy();rng.shuffle(primary)
 return batches,np.array_split(primary,len(batches))

for epoch in range(first,config['epochs']+1):
 torch.manual_seed(3000+epoch);torch.cuda.manual_seed_all(3000+epoch);model.train()
 abatches,pbatches=batch_indices(epoch);sums=np.zeros(9,np.float64);counts=0
 noise=-np.log(-np.log(np.random.default_rng(8000+epoch).uniform(.000001,.999999,(len(teacher.y),4)))).astype(np.float32)
 tick=time.time()
 for step,(ai,pi) in enumerate(zip(abatches,pbatches)):
  progress=((epoch-1)+step/len(abatches))/config['epochs']
  lr=config['final_learning_rate']+(config['learning_rate']-config['final_learning_rate'])*(.5+.5*math.cos(math.pi*progress))
  for group in opt.param_groups:group['lr']=lr*min(1.,((epoch-1)*len(abatches)+step+1)/100)
  b=teacher.batch(ai);ii=b['train_index'];pp=torch.as_tensor(pi,device='cuda')
  opt.zero_grad(set_to_none=True)
  extra,details=model.teacher_forward(x[ii],b,torch.as_tensor(noise[teacher.fragments[ai,2]],device='cuda'))
  h=model.root(x[pp]);primary=duration_score(model.root_logprob(h),target[pp],model.grid)
  table=model.roads(torch.arange(model.nroads+1,device='cuda'))
  alp=model.attachment_logprob(h,**{k:v[pp] for k,v in bank.items()},road_table=table)
  av=attachment[pp]>=0
  attach=torch.nn.functional.nll_loss(alp[av],attachment[pp][av]) if av.any() else alp.sum()*0
  loss=primary+extra+.1*attach;assert torch.isfinite(loss)
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
  values=[float(primary.detach()),float(details['route']),float(details['local']),float(details['value']),
   float(details['kl']),float(attach.detach()),float(details['route_accuracy']),float(loss.detach()),len(pi)]
  sums+=values;counts+=len(pi)
  if step%250==0:
   row=dict(epoch=epoch,step=step,steps=len(abatches),primary_queries_seen=counts,
    means=(sums[:8]/(step+1)).tolist(),style_mass=details['style_mass'].cpu().tolist(),
    epoch_seconds=time.time()-tick,elapsed_seconds=elapsed_before+time.time()-start)
   write(run/'progreso.json',row);print(json.dumps(row),flush=True)
 assert counts==1023485
 row=dict(epoch=epoch,steps=len(abatches),primary_queries=counts,teacher_tokens_once=True,metrics=(sums[:8]/len(abatches)).tolist(),
  seconds=time.time()-tick,elapsed=elapsed_before+time.time()-start)
 history.append(row);write(run/'historia_entrenamiento.json',history)
 state=model.state_dict();size=sum(v.numel()*v.element_size() for v in state.values())
 assert 'D' not in state and size<50*1024**2
 torch.save(dict(state=state,optimizer=opt.state_dict(),epoch=epoch,history=history,elapsed=row['elapsed']),run/'reanudacion.pt')
 print(json.dumps(dict(stage='epoch_complete',**row)),flush=True)

model.eval();finalfile=run/'modelo_secuencia_epoca6.pt'
torch.save(dict(state=model.state_dict(),config=config,features=names,epoch=6),finalfile)
assert finalfile.stat().st_size<50*1024**2
# Finalroot diagnostic only. No route labels are loaded for Validation.
va,V,vnames=feature_input('validation',read(BEST/'resultado.json'));truth=np.load(DATA/'validation_y.npy')
assert names==vnames and len(truth)==127941
parts=[]
with torch.no_grad():
 for a in range(0,len(V),1024):
  vv=torch.as_tensor(V[a:a+1024],device='cuda')
  parts.append(model.root_quantiles(model.root(vv)).cpu().numpy())
q=np.concatenate(parts);m=metric(truth,q[:,1])
np.savez_compressed(run/'validacion_solo_raiz.npz',row=va['row'],timestamp=va['timestamp'],true=truth,pred=q[:,1],quantiles=q)
result=dict(complete=True,stage='A sequence representation complete; StageB publicrouteforecasts pending',epoch=6,
 root_diagnostic_validation=m,root_diagnostic_not_fullproposed_pipeline=True,
 model_file=str(finalfile),model_sha256=sha(finalfile),state_MiB=finalfile.stat().st_size/1024**2,
 primary_queries_per_epoch=1023485,all_6epochs_fullteacher=True,
 sources={n:sha(run/'fuentes'/n) for n in sources},normalization_sha256=sha(run/'normalizacion_Train.npz'),
 root_predictions_sha256=sha(run/'validacion_solo_raiz.npz'),Test_access=False,
 actual_data_files_accessed=sorted(access),seconds=elapsed_before+time.time()-start)
write(run/'resultado_etapa_A.json',result)
p=run/'reanudacion.pt';assert p.resolve().is_relative_to(run.resolve());p.unlink()
print(json.dumps(result),flush=True)
