"""Matched semantic auxiliary training on the strong duration model, one GPU arm."""
from base_replanteo import *
import torch,argparse,shutil
from inferencia import feature_input
from modelo_mezcla_rutas import make_duration_model
from modelo_aux_solicitud import RequestAuxiliary
from perdida_maestro_probabilistico import expected_teacher_score
p=argparse.ArgumentParser();p.add_argument('--arm',choices=['distill'],required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
access=deny_test();proof=read(OUT/'verificacion_aux_solicitud_cpu.json');assert proof['passed']
assert proof['sources']['modelo_aux_solicitud.py']==sha(OUT/'modelo_aux_solicitud.py')
weight=0.;teacher_weight=.2;run=OUT/'alumno_maestro_s0'
config=dict(arm=a.arm,weight=weight,epochs=4,batch=512,seed=20261122,duration_lr=1e-5,request_lr=.001,weight_decay=.0003,geometry_weight=.2,rps_weight=5.,canonical_inference_batch=2048,precision='FP32 highest, no autocast',parent_sha256=sha(BEST/'modelo.pt'),query=['o_lon','o_lat','d_lon','d_lat','timestamp'],Test='sealed',primary_train_rule='1023485 embargoe-safe rows, unchanged evaluation cohort')
teacher_run=OUT/'maestro_cruzado_s0'
teacher_record=read(teacher_run/'resultado.json')
assert teacher_record['all_eligible_rows_assigned_once'] and teacher_record['own_whole_fold_excluded_for_every_row'] and teacher_record['Train_n']==1023485
teacher_file=teacher_run/'parametros_train_fuera_bloque.npz'
assert sha(teacher_file)==teacher_record['parameters_sha256']
teacher_proof=read(OUT/'verificacion_perdida_maestro.json');assert teacher_proof['passed']
assert sha(OUT/'perdida_maestro_probabilistico.py')==teacher_proof['sources']['perdida_maestro_probabilistico.py']
control_record=read(OUT/'aux_solicitud_control_s0/resultado.json')
for key in ['epochs','batch','seed','duration_lr','request_lr','weight_decay','geometry_weight','rps_weight','canonical_inference_batch','precision','parent_sha256','query','primary_train_rule']:
 assert config[key]==control_record['config'][key],('Unmatched control',key)
assert control_record['config']['weight']==0
config.update(teacher_weight=teacher_weight,teacher_parameters_sha256=sha(teacher_file),teacher_parameter_file=str(teacher_file.relative_to(ROOT)),soft_target_probability_sha256=sha(OUT/'perdida_maestro_probabilistico.py'),control_source_sha256=sha(OUT/'entrenar_aux_solicitud.py'),matched_control='aux_solicitud_control_s0',query_uses_teacher=False)
sources=['entrenar_alumno_maestro.py','modelo_aux_solicitud.py','base_replanteo.py','perdida_maestro_probabilistico.py']
if not a.resume:
 assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
 for f in sources:shutil.copy2(OUT/f,run/'fuentes'/f)
 write(run/'protocolo.json',config)
 write(run/'fuentes/parent_sources.json',{str(f.relative_to(FROZEN)):sha(f) for f in FROZEN.rglob('*.py')})
else:
 assert read(run/'protocolo.json')==config
 for f in sources:assert sha(OUT/f)==sha(run/'fuentes'/f)
start=time.time();torch.set_num_threads(4);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True);torch.cuda.set_per_process_memory_fraction(.4)
rec=read(BEST/'resultado.json');ck=torch.load(BEST/'modelo.pt',map_location='cpu',weights_only=False)
duration=make_duration_model(ck,rec['config']);duration.load_state_dict(ck['state']);model=RequestAuxiliary(duration).cuda()
tr,X,names=feature_input('train',rec);va,V,nnames=feature_input('validation',rec)
assert names==nnames==rec['features'] and all(not n.startswith('call_') for n in names)
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index'];assert len(eligible)==1023485
y=np.load(DATA/'train_y.npy');v=np.load(DATA/'validation_y.npy');assert len(v)==127941
soft=dict(np.load(teacher_file))
assert np.array_equal(soft['trip_train_index'],eligible) and np.array_equal(soft['row'],tr['row'][eligible])
assert np.array_equal(soft['timestamp'],tr['timestamp'][eligible])
assert np.array_equal(soft['fold'],((tr['timestamp'][eligible]//(7*86400))%5).astype(np.uint8))
mu_array=np.full(len(y),np.nan,np.float32);ls_array=np.full(len(y),np.nan,np.float32)
mu_array[eligible]=soft['mu_log_seconds'];ls_array[eligible]=soft['log_sigma']
assert np.isfinite(mu_array[eligible]).all() and np.isfinite(ls_array[eligible]).all()
calls=np.searchsorted(np.array(['A','B','C']),tr['call']).astype(np.int64);assert np.isin(tr['call'],['A','B','C']).all()
gt=dict(np.load(DATA/'geometria_train.npz'));pos=np.searchsorted(gt['row'],tr['row']);safe=np.minimum(pos,len(gt['row'])-1)
valid=(pos<len(gt['row']))&(gt['row'][safe]==tr['row'])&(gt['X'][safe,-1]<=500)
geometry=gt['X'][safe,1:15].copy();valid&=np.isfinite(geometry).all(1);geometry[~valid]=0
x=torch.tensor(X,device='cuda');vx=torch.tensor(V,device='cuda');target=torch.tensor(y,device='cuda');c=torch.tensor(calls,device='cuda')
g=torch.tensor(geometry,device='cuda');mask=torch.tensor(valid,device='cuda',dtype=torch.float32);ids=torch.tensor(eligible,device='cuda')
tmu=torch.tensor(mu_array,device='cuda');tls=torch.tensor(ls_array,device='cuda')
del X,V,gt,geometry,calls,soft,mu_array,ls_array
opt=torch.optim.AdamW([dict(params=model.duration.parameters(),lr=config['duration_lr']),dict(params=model.request_head.parameters(),lr=config['request_lr'])],weight_decay=config['weight_decay'])
def objective(index):
 out,q=model(x[index]);t=target[index];label=((t-60)/15).long()
 nll=-out[0].gather(-1,label[:,None,None].expand(-1,16,1)).mean()
 desired=(model.duration.duration_grid[None,None,:]>=t[:,None,None]).float()
 rps=(out[0].exp().cumsum(-1)-desired).square().mean()
 geom=model.duration.geometry_score(out,t,g[index],mask[index])
 request=model.request_score(out,q,t,c[index])
 soft_score=expected_teacher_score(out[0],tmu[index],tls[index])
 return nll+5*rps+.2*geom+weight*request+teacher_weight*soft_score,soft_score
@torch.no_grad()
def predict():
 model.eval();parts=[]
 for block in vx.split(2048):
  out,q=model(block);parts.append(model.duration.quantiles(out[0])[:,1].cpu().numpy())
 return np.concatenate(parts)
protected=dict(np.load(BEST/'validacion.npz'));assert np.array_equal(protected['row'],va['row']) and np.array_equal(protected['true'],v)
history=[];best=rec['validation']['mae_s'];best_epoch=0;best_file=str(BEST/'modelo.pt');all_pred={'epoch_0':protected['pred']};saved=[];first=1
if not a.resume:
 model.eval();opt.zero_grad(set_to_none=True);loss,rq=objective(ids[:32]);assert torch.isfinite(loss);loss.backward()
 assert all(torch.isfinite(t.grad).all() for t in model.parameters() if t.grad is not None)
 assert float(model.request_head.weight.grad.norm())>0 if weight else True
 opt.zero_grad(set_to_none=True)
 check=predict();assert np.array_equal(check,protected['pred'])
 write(run/'verificacion_GPU_inicial.json',dict(passed=True,finite_gradients=True,all_reference_predictions_exact=True,n=len(v),initial_auxiliary_request_NLL=float(rq.detach()),Test_access=False))
 print(json.dumps(dict(stage='initial_GPU_gate',passed=True,arm=a.arm,reference_exact=True)),flush=True)
else:
 resume=torch.load(run/'ultima_reanudacion.pt',map_location='cuda',weights_only=False)
 model.load_state_dict(resume['state']);opt.load_state_dict(resume['optimizer']);history=resume['history'];best=resume['best'];best_epoch=resume['best_epoch'];best_file=resume['best_file'];saved=resume['saved'];first=resume['epoch']+1
 torch.set_rng_state(resume['torch_rng'].cpu());torch.cuda.set_rng_state(resume['cuda_rng'].cpu())
 all_pred=dict(np.load(run/'predicciones_avance.npz'))
for epoch in range(first,5):
 model.train();total=0.;aux=0.;count=0
 for selected in ids[torch.randperm(len(ids),device='cuda')].split(512):
  opt.zero_grad(set_to_none=True);loss,rq=objective(selected);assert torch.isfinite(loss);loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
  n=len(selected);total+=float(loss.detach())*n;aux+=float(rq.detach())*n;count+=n
 prediction=predict();all_pred['epoch_'+str(epoch)]=prediction;m=metric(v,prediction)
 improve=m['mae_s']<best
 preserve=improve or m['mae_s']<131.06
 path=run/('epoca_'+str(epoch)+'.pt')
 if preserve:
  torch.save(dict(state=model.state_dict(),config=config,original_constructor={k:ck[k] for k in ['mean','std','keep','bins','anchors','anchor_indices','geometry_indices']},parent_config=rec['config'],features=names,epoch=epoch),path)
  saved.append(dict(epoch=epoch,file=path.name,sha256=sha(path),validation=m,threshold_qualifies=m['mae_s']<131.06))
 if improve:best=m['mae_s'];best_epoch=epoch;best_file=str(path)
 row=dict(epoch=epoch,train_n=count,train_loss=total/count,teacher_expected_score=aux/count,validation=m,preserved=preserve,seconds=time.time()-start);history.append(row);print(json.dumps(row),flush=True)
 write(run/'historia.json',history);np.savez_compressed(run/'predicciones_avance.npz',**all_pred)
 torch.save(dict(state=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,history=history,best=best,best_epoch=best_epoch,best_file=best_file,saved=saved,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),run/'ultima_reanudacion.pt')
best_pred=all_pred['epoch_'+str(best_epoch)]
np.savez_compressed(run/'validacion.npz',row=va['row'],timestamp=va['timestamp'],true=v,pred=best_pred,**all_pred)
r=dict(id=run.name,config=config,validation=metric(v,best_pred),all_epoch_validation=history,best_epoch=best_epoch,learned_update_selected=best_epoch>0,model_file=best_file,model_sha256=sha(best_file),saved_checkpoints=saved,paired_vs_reference=paired(v,best_pred,protected['pred'],va['timestamp']),source_sha256={f:sha(run/'fuentes'/f) for f in sources},predictions_sha256=sha(run/'validacion.npz'),Train_n=len(eligible),Test_access=False,actual_files_accessed=sorted(access),parent_unchanged=sha(BEST/'modelo.pt')==config['parent_sha256'],seconds=time.time()-start,goal='active unless confirmation target later genuinely achieved')
assert r['parent_unchanged'];write(run/'resultado.json',r)
for file in ['ultima_reanudacion.pt','predicciones_avance.npz']:
 path=run/file
 if path.exists():assert path.resolve().is_relative_to(OUT.resolve());path.unlink()
print(json.dumps(r,default=lambda z:z.tolist() if isinstance(z,np.ndarray) else str(z)),flush=True)
