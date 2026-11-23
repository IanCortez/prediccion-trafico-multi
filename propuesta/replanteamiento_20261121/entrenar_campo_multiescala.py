"""Fresh spatial-field model, with a conditional matched-capacity control."""
from base_replanteo import *
import torch,argparse,shutil
from modelo_campo_multiescala import SpatialFieldJoint
from datos_campo_multiescala import load_data
from modelo_mezcla_rutas import make_duration_model
ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['spatial','capacity'],default='spatial');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
access=deny_test();proof=read(OUT/'verificacion_campo_multiescala.json')
assert proof['passed'] and proof['source_sha256']==sha(OUT/'modelo_campo_multiescala.py')
run=OUT/('campo_multiescala_s0' if args.mode=='spatial' else 'control_capacidad_campo_s0')
config=dict(mode=args.mode,seed=0,max_epochs=50,min_epochs=20,patience=12,meaningful_progress_s=.01,max_training_seconds=3600,batch=512,k=16,width=512,
 lr=.001,weight_decay=.0003,clip=5,geometry_weight=.2,rps_weight=5,canonical_inference_batch=2048,
 precision='FP32 highest, no autocast, standard seeded CUDA',
 initialization='Global mixture and added branch trained from scratch. Only input preprocessing and physical anchors fitted on eligible Train.',
 field='Six bilinear endpoint grids1000,500,250,125,62.5,31.25m,8channels, origin/destination/product plus8calendar signals;128SiLU head adds248 raw parameters shared across16members.',
 confidence='Coordinate-only Train endpoint counts, n/(n+20), zero at unsupported grid nodes. No query path/type/duration/count input.',
 capacity_control='Shared global tower plus1220x1220 coordinate-free dense branch, parameter count close to field; conditional execution only if field beats reference.',
 query=['o_lon','o_lat','d_lon','d_lat','timestamp'],Train_n=1023485,Validation_n=127941,Test='sealed',
 preservation='Every epoch below131.06 and every strict new best below original reference retains checkpoint, plus all point predictions and exact sources.')
sources=['entrenar_campo_multiescala.py','modelo_campo_multiescala.py','datos_campo_multiescala.py','base_replanteo.py']
if not args.resume:
 assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
 for f in sources:shutil.copy2(OUT/f,run/'fuentes'/f)
 write(run/'protocolo.json',config);write(run/'fuentes/parent_sources.json',{str(p.relative_to(FROZEN)):sha(p) for p in FROZEN.rglob('*.py')})
else:
 assert read(run/'protocolo.json')==config
 for f in sources:assert sha(OUT/f)==sha(run/'fuentes'/f)
start=time.time();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(False)
torch.cuda.set_per_process_memory_fraction(.4)
stats_from=OUT/'campo_multiescala_s0/constructor.pt' if args.mode=='capacity' else run/'constructor.pt' if args.resume else None
rec,tr,eligible,train_row,X,y,geometry,valid,va,V,v,ctor=load_data(stats_from)
if not args.resume:torch.save(ctor,run/'constructor.pt')
torch.manual_seed(0);torch.cuda.manual_seed_all(0)
model=SpatialFieldJoint(**{k:ctor[k] for k in ['mean','std','keep','bins','anchors','anchor_indices','geometry_indices','clock_indices','k','width']},mode=args.mode)
h=hashlib.sha256()
for name,t in sorted(model.state_dict().items()):
 if not name.startswith('field.'):h.update(name.encode());h.update(t.detach().cpu().numpy().tobytes())
initial_global_sha=h.hexdigest()
if args.mode=='capacity':
 treatment=read(OUT/'campo_multiescala_s0/resultado.json')
 assert initial_global_sha==treatment['initial_global_state_sha256']
 assert abs(sum(p.numel() for p in model.parameters())/treatment['parameters']-1)<.02
confidence=None
if args.mode=='spatial':
 confidence=model.field.fit_confidence(torch.from_numpy(X[:,:4].copy()))
 write(run/'confianza_coordenadas_Train.json',dict(Train_n=len(X),only_coordinate_fit=True,levels=confidence))
model=model.cuda();x=torch.tensor(X,device='cuda');vx=torch.tensor(V,device='cuda');target=torch.tensor(y,device='cuda')
g=torch.tensor(geometry,device='cuda');mask=torch.tensor(valid,device='cuda',dtype=torch.float32)
del X,V,geometry
opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0003)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=5,min_lr=1e-5)
def objective(index):
 out=model(x[index]);t=target[index];label=((t-60)/15).long()
 nll=-out[0].gather(-1,label[:,None,None].expand(-1,16,1)).mean()
 desired=(model.duration_grid[None,None,:]>=t[:,None,None]).float()
 rps=(out[0].exp().cumsum(-1)-desired).square().mean()
 shape=model.geometry_score(out,t,g[index],mask[index])
 return nll+5*rps+.2*shape
@torch.no_grad()
def predict(m):
 m.eval();result=[]
 for block in vx.split(2048):result.append(m.quantiles(m(block)[0])[:,1].cpu().numpy())
 return np.concatenate(result)
ref=dict(np.load(BEST/'validacion.npz'))
assert np.array_equal(ref['row'],va['row']) and np.array_equal(ref['true'],v)
if not args.resume:
 # This checks the unchanged benchmark separately from the fresh model initialization.
 original=torch.load(BEST/'modelo.pt',map_location='cpu',weights_only=False)
 frozen=make_duration_model(original,rec['config']).cuda();frozen.load_state_dict(original['state'])
 reproduced=predict(frozen);assert np.array_equal(reproduced,ref['pred'])
 del frozen,original;torch.cuda.empty_cache()
 model.eval();opt.zero_grad(set_to_none=True);loss=objective(torch.arange(32,device='cuda'));assert torch.isfinite(loss);loss.backward()
 assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 last=model.field.net[-1] if args.mode=='spatial' else model.field[-1]
 assert float(last.weight.grad.norm())>0
 opt.zero_grad(set_to_none=True)
 write(run/'verificacion_GPU_inicial.json',dict(passed=True,fresh_model_finite_gradients=True,added_branch_output_gradient_nonzero=True,protected_reference_all127941_exact=True,initial_global_state_sha256=initial_global_sha,query_inputs_unchanged=True,Test_access=False))
 print(json.dumps(dict(stage='fresh_field_GPU_gate',mode=args.mode,passed=True,parameters=sum(p.numel() for p in model.parameters()))),flush=True)
torch.manual_seed(0);torch.cuda.manual_seed_all(0)
history=[];all_pred={};best=float('inf');best_epoch=None;best_prediction=None;best_file=None;early_best=float('inf');bad=0;saved=[];first=1;previous_seconds=0.;finished=False
if args.resume:
 checkpoint=torch.load(run/'ultima_reanudacion.pt',map_location='cuda',weights_only=False)
 model.load_state_dict(checkpoint['state']);opt.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['scheduler'])
 history=checkpoint['history'];best=checkpoint['best'];best_epoch=checkpoint['best_epoch'];best_file=checkpoint['best_file'];early_best=checkpoint['early_best'];bad=checkpoint['bad'];saved=checkpoint['saved'];first=checkpoint['epoch']+1;previous_seconds=checkpoint['elapsed'];finished=checkpoint['finished']
 torch.set_rng_state(checkpoint['torch_rng'].cpu());torch.cuda.set_rng_state(checkpoint['cuda_rng'].cpu())
 all_pred=dict(np.load(run/'predicciones_avance.npz'));best_prediction=all_pred['epoch_'+str(best_epoch)];del checkpoint
def save_resume(epoch,finished):
 temp=run/'reanudacion.tmp'
 torch.save(dict(state=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,history=history,best=best,best_epoch=best_epoch,best_file=best_file,early_best=early_best,bad=bad,saved=saved,elapsed=previous_seconds+time.time()-start,finished=finished,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),temp)
 os.replace(temp,run/'ultima_reanudacion.pt')
for epoch in ([] if finished else range(first,51)):
 model.train();total=0.;count=0
 for batch_number,ix in enumerate(torch.randperm(len(x),device='cuda').split(512)):
  opt.zero_grad(set_to_none=True);loss=objective(ix);assert torch.isfinite(loss);loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
  if args.mode=='spatial' and epoch==1 and batch_number==1:
   norm=sum(float(level.embedding.weight.grad.norm()) for level in list(model.field.origin)+list(model.field.destination))
   assert norm>0;print(json.dumps(dict(stage='field_embeddings_learn',gradient_norm=norm)),flush=True)
  opt.step();n=len(ix);total+=float(loss.detach())*n;count+=n
 prediction=predict(model);m=metric(v,prediction);all_pred['epoch_'+str(epoch)]=prediction
 improved=m['mae_s']<best
 if improved:best=m['mae_s'];best_epoch=epoch;best_prediction=prediction
 if m['mae_s']<early_best-.01:early_best=m['mae_s'];bad=0
 else:bad+=1
 preserve=m['mae_s']<131.06 or (improved and m['mae_s']<rec['validation']['mae_s'])
 path=run/('epoca_'+str(epoch)+'.pt')
 if preserve:
  torch.save(dict(state=model.state_dict(),constructor=ctor,mode=args.mode,config=config,epoch=epoch,features=ctor['features']),path)
  saved.append(dict(epoch=epoch,file=path.name,sha256=sha(path),validation=m,threshold_qualifies=m['mae_s']<131.06))
 if improved:best_file=str(path) if preserve else None
 scheduler.step(m['mae_s'])
 row=dict(epoch=epoch,train_n=count,train_loss=total/count,validation=m,preserved=preserve,lr=opt.param_groups[0]['lr'],seconds=previous_seconds+time.time()-start)
 history.append(row);write(run/'historia.json',history);np.savez_compressed(run/'predicciones_avance.npz',**all_pred)
 print(json.dumps(row),flush=True);save_resume(epoch,False)
 if (epoch>=20 and bad>=12) or previous_seconds+time.time()-start>3600:break
assert best_prediction is not None
last_epoch=history[-1]['epoch'];save_resume(last_epoch,True)
np.savez_compressed(run/'validacion.npz',row=va['row'],timestamp=va['timestamp'],true=v,pred=best_prediction,**all_pred)
r=dict(id=run.name,config=config,validation=metric(v,best_prediction),best_epoch=best_epoch,epochs_run=last_epoch,all_epoch_validation=history,
 model_file=best_file,model_sha256=sha(best_file) if best_file else None,weights_retained=bool(best_file),
 saved_checkpoints=saved,parameters=sum(p.numel() for p in model.parameters()),initial_global_state_sha256=initial_global_sha,
 constructor_sha256=sha(run/'constructor.pt'),features=ctor['features'],reference_sha256=rec['model_sha256'],reference_validation=rec['validation'],
 paired_vs_reference=paired(v,best_prediction,ref['pred'],va['timestamp']),
 source_sha256={f:sha(run/'fuentes'/f) for f in sources},predictions_sha256=sha(run/'validacion.npz'),
 fresh_global_and_branch_training=True,Train_n=len(y),Test_access=False,actual_files_accessed=sorted(access),
 reference_unchanged=sha(BEST/'modelo.pt')==rec['model_sha256'],GPU_peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,
 elapsed_seconds=previous_seconds+time.time()-start,goal='active unless target is later genuinely confirmed')
assert r['reference_unchanged'];write(run/'resultado.json',r)
for name in ['ultima_reanudacion.pt','predicciones_avance.npz']:
 path=run/name;assert path.resolve().is_relative_to(OUT.resolve());path.unlink()
print(json.dumps(dict(id=run.name,validation=r['validation'],best_epoch=best_epoch,epochs=last_epoch,saved_checkpoints=saved,seconds=r['elapsed_seconds'])),flush=True)
