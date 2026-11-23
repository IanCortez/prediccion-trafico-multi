"""Predeclared exact subtrip auxiliary task; original OD-duration head at inference."""
from base_replanteo import *
import torch,argparse,shutil
from inferencia import feature_input
from modelo_mezcla_rutas import make_duration_model
from modelo_aux_subviajes import SubtripAuxiliary,duration_score
ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['auxiliary','control'],default='auxiliary');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
access=deny_test();proof=read(OUT/'verificacion_aux_subviajes.json')
assert proof['passed'] and proof['model_source_sha256']==sha(OUT/'modelo_aux_subviajes.py')
data=OUT/'subviajes_train_s0';datarec=read(data/'resultado.json')
assert datarec['exact_interval_labels'] and datarec['original_cohort_unchanged']
ctor=torch.load(data/'constructor_aux.pt',map_location='cpu',weights_only=False)
assert proof['constructor_sha256']==sha(data/'constructor_aux.pt')
run=OUT/('aux_subviajes_s0' if args.arm=='auxiliary' else 'control_aux_subviajes_s0')
config=dict(arm=args.arm,seed=20261122,joint_epochs=6,auxiliary_warmup_epochs=1,batch=512,
 primary_lr=1e-5,auxiliary_lr=.001,weight_decay=.0003,auxiliary_weight=.2 if args.arm=='auxiliary' else 0.,
 geometry_weight=.2,rps_weight=5,clip=5,precision='FP32 highest, standard seeded CUDA; no bitwise training guarantee',
 primary_Train_n=1023485,aux_Train_n=datarec['aux_examples'],Validation_n=127941,
 sampling='All primary rows once per epoch;512 random auxiliary rows per primary step, separate seeded RNG. Same auxiliary forwards in conditional control to match dropout draws.',
 warmup='One auxiliary epoch with shared duration model frozen/eval; original full validation predictions must remain exact.',
 initialization='Protected original best duration model; independent per-member8component auxiliary duration head. Primary head only at query.',
 query=['o_lon','o_lat','d_lon','d_lat','timestamp'],Test='sealed',canonical_inference_batch=2048,
 parent_sha256=sha(BEST/'modelo.pt'),aux_features_sha256=datarec['outputs']['features_train.npz']['sha256'],
 preserve='All metrics/predictions including MAE>131.06; weights for every strict primary new best and complete MAE<131.06 candidate.')
sources=['entrenar_aux_subviajes.py','modelo_aux_subviajes.py','consulta_subviajes_train.py','consulta_historicos_cruzados.py','base_replanteo.py']
if not args.resume:
 assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
 for s in sources:shutil.copy2(OUT/s,run/'fuentes'/s)
 write(run/'protocolo.json',config)
 write(run/'fuentes/parent_sources.json',{str(p.relative_to(FROZEN)):sha(p) for p in FROZEN.rglob('*.py')})
else:
 assert read(run/'protocolo.json')==config
 for s in sources:assert sha(OUT/s)==sha(run/'fuentes'/s)
start=time.time();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(False)
torch.cuda.set_per_process_memory_fraction(.4);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
rec=read(BEST/'resultado.json');ck=torch.load(BEST/'modelo.pt',map_location='cpu',weights_only=False)
duration=make_duration_model(ck,rec['config']);duration.load_state_dict(ck['state'])
model=SubtripAuxiliary(duration,**ctor).cuda()
tr,X,names=feature_input('train',rec);va,V,vnames=feature_input('validation',rec)
assert names==vnames==rec['features']==datarec['features']
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index'];assert len(eligible)==1023485
y=np.load(DATA/'train_y.npy');v=np.load(DATA/'validation_y.npy')
gt=dict(np.load(DATA/'geometria_train.npz'));pos=np.searchsorted(gt['row'],tr['row']);safe=np.minimum(pos,len(gt['row'])-1)
valid=(pos<len(gt['row']))&(gt['row'][safe]==tr['row'])&(gt['X'][safe,-1]<=500)
geometry=gt['X'][safe,1:15].copy();valid&=np.isfinite(geometry).all(1);geometry[~valid]=0
A=np.load(data/'features_train.npz')['X'];ay=np.load(data/'consultas_etiquetas_procedencia.npz')['y'].astype(np.float32)
assert sha(data/'features_train.npz')==config['aux_features_sha256']
x=torch.tensor(X,device='cuda');vx=torch.tensor(V,device='cuda');target=torch.tensor(y,device='cuda')
ax=torch.tensor(A,device='cuda');at=torch.tensor(ay,device='cuda')
g=torch.tensor(geometry,device='cuda');mask=torch.tensor(valid,device='cuda',dtype=torch.float32);ids=torch.tensor(eligible,device='cuda')
del X,V,A,ay,geometry,gt
ref=dict(np.load(BEST/'validacion.npz'));assert np.array_equal(ref['row'],va['row']) and np.array_equal(ref['true'],v)
@torch.no_grad()
def predict():
 model.eval();parts=[]
 for block in vx.split(2048):parts.append(model.duration.quantiles(model(block)[0])[:,1].cpu().numpy())
 return np.concatenate(parts)
def primary_score(i):
 out=model(x[i]);t=target[i]
 return duration_score(out[0],t,model.duration.duration_grid)+.2*model.duration.geometry_score(out,t,g[i],mask[i])
history=[];saved=[];all_pred={'epoch_0':ref['pred']};best=rec['validation']['mae_s'];best_epoch=0;best_file=str(BEST/'modelo.pt');first=1;warm=None;previous_seconds=0.
if not args.resume:
 model.eval();check=predict();assert np.array_equal(check,ref['pred'])
 model.zero_grad(set_to_none=True)
 loss=primary_score(ids[:32])+.2*duration_score(model.auxiliary(ax[:32]),at[:32],model.duration.duration_grid)
 assert torch.isfinite(loss);loss.backward()
 assert all(torch.isfinite(z.grad).all() for z in model.parameters() if z.grad is not None) and model.aux_weight.grad.norm()>0
 model.zero_grad(set_to_none=True)
 write(run/'verificacion_GPU_inicial.json',dict(passed=True,full_reference_prediction_exact=True,n=len(v),all_gradients_finite=True,query_unchanged=True,Test_access=False))
 print(json.dumps(dict(stage='subtrip_GPU_gate',passed=True,reference_exact=True)),flush=True)
 # Fit only the task head before exposing the shared encoder to its gradient.
 for z in model.duration.parameters():z.requires_grad_(False)
 model.eval();warm_opt=torch.optim.AdamW(model.aux_parameters(),lr=.001,weight_decay=.0003)
 order=torch.randperm(len(ax),device='cuda',generator=torch.Generator(device='cuda').manual_seed(config['seed']+501))
 total=0.;count=0
 for i in order.split(512):
  warm_opt.zero_grad(set_to_none=True);loss=duration_score(model.auxiliary(ax[i]),at[i],model.duration.duration_grid)
  assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.aux_parameters(),5,error_if_nonfinite=True);warm_opt.step()
  total+=float(loss.detach())*len(i);count+=len(i)
 for z in model.duration.parameters():z.requires_grad_(True)
 check=predict();assert np.array_equal(check,ref['pred'])
 warm=dict(aux_examples=count,auxiliary_loss=total/count,primary_all127941_predictions_still_exact=True,seconds=time.time()-start)
 write(run/'calentamiento.json',warm);print(json.dumps(dict(stage='auxiliary_head_warmup',**warm)),flush=True)
 del warm_opt
 opt=torch.optim.AdamW([dict(params=model.duration.parameters(),lr=1e-5),dict(params=model.aux_parameters(),lr=.001)],weight_decay=.0003)
 torch.save(dict(state=model.state_dict(),optimizer=opt.state_dict(),epoch=0,history=history,saved=saved,best=best,best_epoch=best_epoch,best_file=best_file,warm=warm,elapsed=time.time()-start),run/'ultima_reanudacion.pt')
else:
 state=torch.load(run/'ultima_reanudacion.pt',map_location='cuda',weights_only=False)
 model.load_state_dict(state['state'])
 opt=torch.optim.AdamW([dict(params=model.duration.parameters(),lr=1e-5),dict(params=model.aux_parameters(),lr=.001)],weight_decay=.0003);opt.load_state_dict(state['optimizer'])
 first=state['epoch']+1;history=state['history'];saved=state['saved'];best=state['best'];best_epoch=state['best_epoch'];best_file=state['best_file'];warm=state['warm'];previous_seconds=state['elapsed']
 if (run/'predicciones_avance.npz').exists():all_pred=dict(np.load(run/'predicciones_avance.npz'))
for epoch in range(first,7):
 # Explicit epoch seeds permit exact continuation and matched primary/auxiliary sampling.
 torch.manual_seed(config['seed']+epoch);torch.cuda.manual_seed_all(config['seed']+epoch);model.train()
 order=ids[torch.randperm(len(ids),device='cuda',generator=torch.Generator(device='cuda').manual_seed(config['seed']+1000+epoch))]
 auxrng=torch.Generator(device='cuda').manual_seed(config['seed']+2000+epoch)
 total=0.;aux_total=0.;count=0
 for i in order.split(512):
  ai=torch.randint(len(ax),(len(i),),device='cuda',generator=auxrng)
  opt.zero_grad(set_to_none=True);main=primary_score(i);aux=duration_score(model.auxiliary(ax[ai]),at[ai],model.duration.duration_grid)
  loss=main+config['auxiliary_weight']*aux;assert torch.isfinite(loss);loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
  total+=float(main.detach())*len(i);aux_total+=float(aux.detach())*len(i);count+=len(i)
 pred=predict();all_pred['epoch_'+str(epoch)]=pred;m=metric(v,pred);improved=m['mae_s']<best;preserve=improved or m['mae_s']<131.06
 path=run/('epoca_'+str(epoch)+'.pt')
 if preserve:
  torch.save(dict(state=model.state_dict(),config=config,aux_constructor=ctor,parent_config=rec['config'],
   original_constructor={k:ck[k] for k in ['mean','std','keep','bins','anchors','anchor_indices','geometry_indices']},features=names,epoch=epoch),path)
  saved.append(dict(epoch=epoch,file=path.name,sha256=sha(path),validation=m,threshold_qualifies=m['mae_s']<131.06))
 if improved:best=m['mae_s'];best_epoch=epoch;best_file=str(path)
 row=dict(epoch=epoch,primary_train_n=count,primary_loss=total/count,auxiliary_loss=aux_total/count,validation=m,preserved=preserve,seconds=previous_seconds+time.time()-start)
 history.append(row);write(run/'historia.json',history);print(json.dumps(row),flush=True)
 np.savez_compressed(run/'predicciones_avance.npz',**all_pred)
 torch.save(dict(state=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,history=history,saved=saved,best=best,best_epoch=best_epoch,best_file=best_file,warm=warm,elapsed=previous_seconds+time.time()-start),run/'ultima_reanudacion.pt')
best_pred=all_pred['epoch_'+str(best_epoch)]
np.savez_compressed(run/'validacion.npz',row=va['row'],timestamp=va['timestamp'],true=v,pred=best_pred,**all_pred)
r=dict(id=run.name,config=config,validation=metric(v,best_pred),best_epoch=best_epoch,learned_update_selected=best_epoch>0,all_epoch_validation=history,warmup=warm,
 model_file=best_file,model_sha256=sha(best_file),saved_checkpoints=saved,paired_vs_reference=paired(v,best_pred,ref['pred'],va['timestamp']),
 predictions_sha256=sha(run/'validacion.npz'),sources={s:sha(run/'fuentes'/s) for s in sources},data_manifest_sha256=sha(data/'resultado.json'),
 Test_access=False,actual_data_files_accessed=sorted(access),parent_unchanged=sha(BEST/'modelo.pt')==config['parent_sha256'],
 seconds=previous_seconds+time.time()-start,goal='Still active unless strong frozen candidate later achieves confirmation target')
assert r['parent_unchanged'];write(run/'resultado.json',r)
for name in ['ultima_reanudacion.pt','predicciones_avance.npz']:
 path=run/name
 if path.exists():assert path.resolve().is_relative_to(OUT.resolve());path.unlink()
print(json.dumps(dict(stage='complete',id=r['id'],validation=r['validation'],best_epoch=best_epoch,saved_checkpoints=len(saved),seconds=r['seconds'])),flush=True)
