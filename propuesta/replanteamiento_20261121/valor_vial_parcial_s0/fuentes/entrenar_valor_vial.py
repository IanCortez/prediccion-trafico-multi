"""Fresh OD street-value distribution with masked Train-only road progress auxiliaries."""
from base_replanteo import *
import torch,shutil,gc,argparse
from inferencia import feature_input
from modelo_valor_vial import RoadValueNet
from modelo_aux_subviajes import duration_score
ap=argparse.ArgumentParser();ap.add_argument('--control',action='store_true');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
access=deny_test();start=time.time();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction(.5)
torch.manual_seed(20261122);torch.cuda.manual_seed_all(20261122)
proof=read(OUT/'verificacion_valor_vial_cpu.json');assert proof['passed'] and proof['model_sha256']==sha(OUT/'modelo_valor_vial.py')
folder=OUT/'banco_valor_vial_s0';bankmeta=read(folder/'resultado.json');auxfolder=OUT/'estados_valor_vial_s0';auxmeta=read(auxfolder/'resultado.json')
assert auxmeta['complete'];assert sha(auxfolder/'estados_Train.npz')==auxmeta['targets_sha256']
reference=read(OUT/'nuevo_mejor_verificado.json')
config=dict(mode='control' if args.control else 'partial_street_auxiliary',seed=20261122,max_epochs=40,max_training_seconds=3600,minimum_epochs=12,patience=8,
 batch=512,lr=.0007,weight_decay=.0003,clip=5,precision='FP32highest',canonical_inference_batch=2048,GPU_memory_fraction=.5,
 architecture='Fresh4memberamortizedstreetvalue; learned53766roadembeddings withonehopgraphcontext;16uncertainpublicendpointpairs*2timecomponents plus2globalfallback; no route rollout or target teacherforcing inpublicODprediction.',
 loss='PrimaryNLL+5RPS; treatment adds.05continuousremainingtimeNLL+.05partiallocaltimeHuber+.1neighbor/EOSrouteCE.',
 input='Same216legalraw5derivedfeatures as protectedreference, including existingwholeweekcrossfitTraintraffic summaries; primary normalizer uses eligibleTrainonly. Purepublic206 was used only for heldTrainoracle to avoid nestedhistoryleak.',
 auxiliary='FourfixedtrustedTrainstates, one rotatingpertrip/epoch. Actualroad/remainingtime onlyauxmethod; every originalprimaryTrain query retained.',
 bank_manifest_sha256=sha(folder/'resultado.json'),aux_manifest_sha256=sha(auxfolder/'resultado.json'),
 Train_n=1023485,Validation_n=127941,Test='sealed',reference_id=reference['id'],reference_validation=reference['validation'],
 control_trigger='At least.5second fullValidation improvement overprotectedbest and upperdaybootstrapdeltaCI<0, then certifycandidate andmatchednoauxcontrol.',
 preservation='Allcompletedmetrics/predictions; every completeMAE<131.06checkpoint and strictnewbest. Nonqualifyingweights temporaryonly.')
run=OUT/('control_valor_vial_s0' if args.control else 'valor_vial_parcial_s0');sources=['entrenar_valor_vial.py','modelo_valor_vial.py','modelo_aux_subviajes.py','base_replanteo.py']
if args.control:
 assert read(OUT/'valor_vial_parcial_s0/resultado.json')['control_trigger_met']
if not args.resume:
 assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
 for name in sources:shutil.copy2(OUT/name,run/'fuentes'/name)
 write(run/'protocolo.json',config)
else:
 old=read(run/'protocolo.json')
 # The reference may have been promoted after a completed treatment; training reference stays frozen.
 for key in ['reference_id','reference_validation']:config[key]=old[key]
 assert old==config
 for name in sources:assert sha(OUT/name)==sha(run/'fuentes'/name)
tr,X,names=feature_input('train',read(BEST/'resultado.json'));va,V,vnames=feature_input('validation',read(BEST/'resultado.json'));assert names==vnames and len(names)==216
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index'];y=np.load(DATA/'train_y.npy');truth=np.load(DATA/'validation_y.npy')
assert len(eligible)==1023485 and np.max(tr['timestamp'][eligible]+y[eligible])<int(va['timestamp'].min())
normalfile=run/'normalizacion_Train.npz'
if not args.resume:
 mean=X[eligible].mean(0,dtype=np.float64).astype(np.float32);std=X[eligible].std(0,dtype=np.float64).clip(.01).astype(np.float32)
 np.savez_compressed(normalfile,mean=mean,std=std)
else:normal=dict(np.load(normalfile));mean=normal['mean'];std=normal['std']
g=dict(np.load(folder/'grafo_estatico.npz'));assert sha(folder/'grafo_estatico.npz')==bankmeta['static_graph_sha256']
model=RoadValueNet(mean,std,g['static'],g['neighbors'],g['neighbor_valid'])
initial_hash=hashlib.sha256()
for name,value in sorted(model.state_dict().items()):initial_hash.update(name.encode());initial_hash.update(value.numpy().tobytes())
initial_hash=initial_hash.hexdigest()
if args.control:assert initial_hash==read(OUT/'valor_vial_parcial_s0/resultado.json')['initial_weights_sha256']
model=model.cuda();x=torch.tensor(X,device='cuda');vx=torch.tensor(V,device='cuda');del X,V
banks=[]
for split,d in [('train',tr),('validation',va)]:
 file=folder/(split+'_proyecciones.npz');assert sha(file)==bankmeta['outputs'][file.name]['sha256'];z=dict(np.load(file))
 for key in ['row','timestamp']:assert np.array_equal(z[key],d[key])
 banks.append({k:torch.as_tensor(z[k].astype(np.int64) if k=='chain' else z[k],device='cuda') for k in ['chain','fraction','error_m','remaining_km','reachable']})
bank,vbank=banks;del banks,z
a=dict(np.load(auxfolder/'estados_Train.npz'))
aux={key:torch.as_tensor(value.astype(np.int64) if key in ['chain','destination','route_target'] else value,device='cuda') for key,value in a.items()};del a
target=torch.tensor(y,device='cuda');ids=torch.tensor(eligible,device='cuda')
ref_run=OUT/config['reference_id'];ref=dict(np.load(ref_run/'validacion.npz'))
for key in ['row','timestamp']:assert np.array_equal(ref[key],va[key])
assert np.array_equal(ref['true'],truth)
def select(b,index):return {key:value[index] for key,value in b.items()}
def objective(index,slot):
 logp,h=model.public_logprob(x[index],**select(bank,index),return_hidden=True);loss=duration_score(logp,target[index],model.duration_grid)
 if not args.control:
  sampled={key:value[index,slot] for key,value in aux.items()};extra,details=model.auxiliary_loss(h,**sampled);loss=loss+extra
 return loss
@torch.no_grad()
def predict():
 gc.collect();torch.cuda.empty_cache();model.eval();parts=[]
 for first in range(0,len(vx),2048):
  end=min(first+2048,len(vx));n=end-first;xx=vx[first:end];bb={key:value[first:end] for key,value in vbank.items()}
  if n<2048:
   xx=torch.cat([xx,xx[-1:].expand(2048-n,-1)])
   bb={key:torch.cat([value,value[-1:].expand((2048-n,)+value.shape[1:])]) for key,value in bb.items()}
  lp=model.public_logprob(xx,**bb);parts.append(model.quantiles(lp)[:n].cpu().numpy());del lp
 return np.concatenate(parts)
# Real216feature objective and all auxiliary gradient paths checked before primary evaluation.
model.train();pre=objective(ids[:32],0);assert torch.isfinite(pre);pre.backward()
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
assert model.embedding.weight.grad is not None and float(model.embedding.weight.grad.abs().sum())>0
model.zero_grad(set_to_none=True)
model.eval()
with torch.no_grad():
 ii=ids[:17];lp=model.public_logprob(x[ii],**select(bank,ii));reverse=ii.flip(0)
 rp=model.public_logprob(x[reverse],**select(bank,reverse)).flip(0)
 sp=torch.cat([model.public_logprob(x[i:i+1],**select(bank,slice(int(i),int(i)+1))) for i in ii])
 assert torch.equal(model.quantiles(lp),model.quantiles(rp)) and torch.equal(model.quantiles(lp),model.quantiles(sp))
 assert float((lp.exp()-sp.exp()).abs().max())<2e-6
write(run/'verificacion_GPU_inicial.json',dict(passed=True,Train_loss=float(pre.detach()),input_features=216,primary_and_auxiliary_gradients_finite=True,
 learned_road_embedding_gradient_nonzero=True,raw_target_payload_not_a_public_method_argument=True,batch_reverse_singleton_quantiles_exact=True,
 initial_weights_sha256=initial_hash,parameters=sum(p.numel() for p in model.parameters()),Test=False))
opt=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
history=[];saved=[];best=float('inf');best_epoch=None;best_file=None;initial=None;all_pred={};elapsed_before=0;first_epoch=1;finished=False
temporary=run/'mejor_temporal.pt'
def retain(epoch,quantiles):
 global best,best_epoch,best_file
 pred=quantiles[:,1];m=metric(truth,pred);qualifies=m['mae_s']<131.06 or m['mae_s']<config['reference_validation']['mae_s'];file=None
 if qualifies:
  file=run/('epoca_'+str(epoch)+'.pt');torch.save(dict(state=model.state_dict(),config=config,features=names,epoch=epoch,initial_weights_sha256=initial_hash),file)
  saved.append(dict(epoch=epoch,file=file.name,sha256=sha(file),validation=m,threshold_qualifies=m['mae_s']<131.06))
 if m['mae_s']<best:
  best=m['mae_s'];best_epoch=epoch;best_file=str(file) if file else None
  if not file:torch.save(dict(state=model.state_dict(),config=config,features=names,epoch=epoch),temporary)
 write(run/'candidatos_conservados.json',dict(candidates=saved,Test='sealed'))
 return m,qualifies
if args.resume:
 state=torch.load(run/'ultima_reanudacion.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['state']);opt.load_state_dict(state['optimizer'])
 history=state['history'];saved=state['saved'];best=state['best'];best_epoch=state['best_epoch'];best_file=state['best_file'];initial=state['initial']
 elapsed_before=state['elapsed'];first_epoch=state['epoch']+1;finished=state['finished'];all_pred=dict(np.load(run/'predicciones_avance.npz'))
else:
 q=predict();all_pred['epoch_0']=q;initial,_=retain(0,q);print(json.dumps(dict(stage='initial',validation=initial,seconds=time.time()-start)),flush=True)
for epoch in range(first_epoch,config['max_epochs']+1):
 if finished:break
 model.train();torch.manual_seed(1000+epoch);torch.cuda.manual_seed_all(1000+epoch)
 order=ids[torch.randperm(len(ids),device='cuda',generator=torch.Generator(device='cuda').manual_seed(1000+epoch))]
 total=0.;count=0;slot=(epoch-1)%4
 for index in order.split(config['batch']):
  opt.zero_grad(set_to_none=True);loss=objective(index,slot);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
  total+=float(loss.detach())*len(index);count+=len(index)
 q=predict();all_pred['epoch_'+str(epoch)]=q;m,kept=retain(epoch,q);elapsed=elapsed_before+time.time()-start
 finished=epoch>=config['max_epochs'] or elapsed>=config['max_training_seconds'] or (epoch>=config['minimum_epochs'] and epoch-best_epoch>=config['patience'])
 row=dict(epoch=epoch,primary_train_n=count,train_loss=total/count,validation=m,preserved=kept,seconds=elapsed);history.append(row)
 write(run/'historia.json',history);print(json.dumps(row),flush=True)
 np.savez_compressed(run/'predicciones_avance.npz',**all_pred)
 torch.save(dict(state=model.state_dict(),optimizer=opt.state_dict(),history=history,saved=saved,best=best,best_epoch=best_epoch,best_file=best_file,initial=initial,
  elapsed=elapsed,epoch=epoch,finished=finished),run/'ultima_reanudacion.pt')
selected=all_pred['epoch_'+str(best_epoch)];comparison=paired(truth,selected[:,1],ref['pred'],va['timestamp'])
trigger=bool(comparison['delta_mae_s']<=-.5 and comparison['ci95_s'][1]<0)
np.savez_compressed(run/'validacion.npz',row=va['row'],timestamp=va['timestamp'],true=truth,pred=selected[:,1],quantiles=selected,**all_pred)
result=dict(id=run.name,config=config,validation=metric(truth,selected[:,1]),initial_validation=initial,best_epoch=best_epoch,epochs_run=len(history),model_file=best_file,
 model_sha256=sha(best_file) if best_file else None,initial_weights_sha256=initial_hash,paired_vs_reference=comparison,control_trigger_met=trigger,
 saved_checkpoints=saved,normalization_sha256=sha(normalfile),predictions_sha256=sha(run/'validacion.npz'),sources={name:sha(run/'fuentes'/name) for name in sources},
 Test_access=False,actual_data_files_accessed=sorted(access),seconds=elapsed_before+time.time()-start,
 scope='CompleteunchangedprimaryODValidation; learnedstreetvalue does not roll out a teacherforcedroute. ReusedValidation not independentconfirmation.')
write(run/'resultado.json',result)
for name in ['ultima_reanudacion.pt','predicciones_avance.npz','mejor_temporal.pt']:
 file=run/name
 if file.exists():assert file.resolve().is_relative_to(run.resolve());file.unlink()
print(json.dumps(dict(stage='complete',validation=result['validation'],best_epoch=best_epoch,epochs=len(history),saved=len(saved),control_trigger_met=trigger,seconds=result['seconds'])),flush=True)
