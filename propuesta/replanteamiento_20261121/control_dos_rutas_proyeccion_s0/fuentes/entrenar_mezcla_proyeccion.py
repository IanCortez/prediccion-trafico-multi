"""Fixed uncertain-projection mixture study; full primary cohort, one GPU."""
from base_replanteo import *
import torch,argparse,shutil
from inferencia import feature_input
from modelo_mezcla_rutas import make_duration_model
from modelo_aux_subviajes import duration_score
from modelo_mezcla_proyeccion import ProjectionMixture
ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['expanded','defaults'],default='expanded');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
access=deny_test();data=OUT/'banco_proyeccion_incierta_s0';ds=read(data/'resultado.json');proof=read(OUT/'verificacion_mezcla_proyeccion.json')
assert proof['passed'] and proof['model_source_sha256']==sha(OUT/'modelo_mezcla_proyeccion.py') and proof['data_manifest_sha256']==sha(data/'resultado.json')
run=OUT/('mezcla_proyeccion_incierta_s0' if args.mode=='expanded' else 'control_dos_rutas_proyeccion_s0')
config=dict(mode=args.mode,seed=20261122,epochs=9,frozen_base_epochs=3,joint_epochs=6,batch=512,base_lr=1e-5,branch_lr=.001,weight_decay=.0003,clip=5,
 geometry_weight=.2,rps_weight=5,initial_gate_logit=-4,gps_prior_sigma_m=30,
 architecture='Protected8component time/geometry mixture plus8public-route candidate components with a shared545-128-128-31head. Gate574-64-1uses global hidden state and mean/std of normalized candidate set. Extra hypotheses preserve unknown-route base modes.',
 control='Same constructor,state initialization,normalization,optimizer,sampling and loops; only first two protected route candidates available.',
 training='Three epochs frozen protected tower, then six joint epochs. Original primary labels/filters and query information unchanged.',
 query=['o_lon','o_lat','d_lon','d_lat','timestamp'],primary_Train_n=1023485,Validation_n=127941,Test='sealed',
 precision='FP32 highest model; canonicalFP16public-route inputs; standard seeded CUDA, no bitwise training guarantee',
 canonical_inference_batch=2048,candidate_head_inference_chunk=256,
 initialization_note='The default initial new mixture includes a1.8percent candidate gate; its epoch0 is not the original reference. Base-only control must reproduce reference exactly.',
 parent_sha256=sha(BEST/'modelo.pt'),data_manifest_sha256=sha(data/'resultado.json'),
 preserve='All completed metrics/predictions, even above131.06; weights for every strict new primary best and every fullMAE<131.06.')
sources=['entrenar_mezcla_proyeccion.py','modelo_mezcla_proyeccion.py','modelo_aux_subviajes.py','rutas_proyeccion_incierta.py','consulta_historicos_cruzados.py','base_replanteo.py']
if not args.resume:
 assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
 for s in sources:shutil.copy2(OUT/s,run/'fuentes'/s)
 write(run/'protocolo.json',config);write(run/'fuentes/parent_sources.json',{str(p.relative_to(FROZEN)):sha(p) for p in FROZEN.rglob('*.py')})
else:
 assert read(run/'protocolo.json')==config
 for s in sources:assert sha(OUT/s)==sha(run/'fuentes'/s)
start=time.time();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(False)
torch.cuda.set_per_process_memory_fraction(.4);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
rec=read(BEST/'resultado.json');ck=torch.load(BEST/'modelo.pt',map_location='cpu',weights_only=False)
duration=make_duration_model(ck,rec['config']);duration.load_state_dict(ck['state'])
norm=dict(np.load(data/'normalizacion_Train.npz'));assert sha(data/'normalizacion_Train.npz')==ds['normalization_sha256']
model=ProjectionMixture(duration,norm['mean'],norm['std'],args.mode)
h=hashlib.sha256()
for name,p in sorted(model.state_dict().items()):h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
initial_state_sha=h.hexdigest()
if args.mode=='defaults':
 treatment=read(OUT/'mezcla_proyeccion_incierta_s0/resultado.json');assert initial_state_sha==treatment['initial_model_state_sha256']
model=model.cuda()
tr,X,names=feature_input('train',rec);va,V,vnames=feature_input('validation',rec);assert names==vnames==rec['features']
assert names[int(model.duration.anchor_indices[0])]=='mm_length_gps_time_600s' and names[int(model.duration.anchor_indices[1])]=='mm_free_gps_time_600s'
banks=[]
for split,d in [('train',tr),('validation',va)]:
 path=data/(split+'_banco.npz');assert sha(path)==ds['outputs'][path.name]['sha256'];z=dict(np.load(path))
 assert np.array_equal(z['row'],d['row']) and np.array_equal(z['timestamp'],d['timestamp'])
 banks.append(torch.tensor(z['bank'],device='cuda',dtype=torch.float16))
candidates,vcandidates=banks;del banks,z
x=torch.tensor(X,device='cuda');vx=torch.tensor(V,device='cuda')
y=np.load(DATA/'train_y.npy');v=np.load(DATA/'validation_y.npy');target=torch.tensor(y,device='cuda')
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index'];assert len(eligible)==1023485;ids=torch.tensor(eligible,device='cuda')
gt=dict(np.load(DATA/'geometria_train.npz'));pos=np.searchsorted(gt['row'],tr['row']);safe=np.minimum(pos,len(gt['row'])-1)
valid=(pos<len(gt['row']))&(gt['row'][safe]==tr['row'])&(gt['X'][safe,-1]<=500)
geometry=gt['X'][safe,1:15].copy();valid&=np.isfinite(geometry).all(1);geometry[~valid]=0
g=torch.tensor(geometry,device='cuda');mask=torch.tensor(valid,device='cuda',dtype=torch.float32)
del X,V,gt,geometry
ref=dict(np.load(BEST/'validacion.npz'));assert np.array_equal(ref['row'],va['row']) and np.array_equal(ref['true'],v)
@torch.no_grad()
def predict(base_only=False):
 model.eval();parts=[]
 for first in range(0,len(vx),2048):
  out=model.base_only(vx[first:first+2048]) if base_only else model(vx[first:first+2048],vcandidates[first:first+2048])
  parts.append(model.duration.quantiles(out[0])[:,1].cpu().numpy())
 return np.concatenate(parts)
def objective(index):
 out=model(x[index],candidates[index]);t=target[index]
 return duration_score(out[0],t,model.duration.duration_grid)+.2*model.duration.geometry_score(out,t,g[index],mask[index])
opt=torch.optim.AdamW([dict(params=model.duration.parameters(),lr=1e-5),dict(params=model.branch_parameters(),lr=.001)],weight_decay=.0003)
history=[];saved=[];all_pred={'protected_reference':ref['pred']};best=rec['validation']['mae_s'];best_epoch=None;best_file=str(BEST/'modelo.pt');first_epoch=1;previous_seconds=0.;initial_metric=None
def retain(epoch,pred):
 global best,best_epoch,best_file
 m=metric(v,pred);improved=m['mae_s']<best;preserve=improved or m['mae_s']<131.06
 path=run/('epoca_'+str(epoch)+'.pt')
 if preserve:
  torch.save(dict(state=model.state_dict(),config=config,parent_config=rec['config'],
   original_constructor={k:ck[k] for k in ['mean','std','keep','bins','anchors','anchor_indices','geometry_indices']},features=names,epoch=epoch),path)
  saved.append(dict(epoch=epoch,file=path.name,sha256=sha(path),validation=m,threshold_qualifies=m['mae_s']<131.06))
 if improved:best=m['mae_s'];best_epoch=epoch;best_file=str(path)
 return m,preserve
if not args.resume:
 model.eval();assert np.array_equal(predict(base_only=True),ref['pred'])
 opt.zero_grad(set_to_none=True);loss=objective(ids[:32]);assert torch.isfinite(loss);loss.backward()
 assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 assert model.candidate_head[-1].weight.grad.norm()>0 and model.gate[-1].weight.grad.norm()>0
 with torch.no_grad():
  out=model(x[ids[:32]],candidates[ids[:32]]);label=((target[ids[:32]]-60)/15).long()
  logp_at_time=out[0].gather(-1,label[:,None,None].expand(-1,16,1))[...,0]
  component_at_time=out[4].gather(-1,label[:,None,None,None].expand(-1,16,16,1))[...,0]
  posterior_error=float(((out[1]+component_at_time-logp_at_time[...,None]).exp().sum(-1)-1).abs().max())
 assert posterior_error<1e-5;opt.zero_grad(set_to_none=True)
 write(run/'verificacion_GPU_inicial.json',dict(passed=True,base_only_all127941_reference_predictions_exact=True,
  complete_joint_time_geometry_gradients_finite=True,posterior_component_normalization_error=posterior_error,
  initial_new_mixture_differs_from_reference_by_design=True,Test_access=False))
 pred=predict();all_pred['epoch_0']=pred;initial_metric,preserved=retain(0,pred)
 print(json.dumps(dict(stage='initial_new_mixture',validation=initial_metric,preserved=preserved,base_only_reference_exact=True,parameters=sum(p.numel() for p in model.parameters()))),flush=True)
else:
 state=torch.load(run/'ultima_reanudacion.pt',map_location='cuda',weights_only=False);model.load_state_dict(state['state']);opt.load_state_dict(state['optimizer'])
 history=state['history'];saved=state['saved'];best=state['best'];best_epoch=state['best_epoch'];best_file=state['best_file'];first_epoch=state['epoch']+1;previous_seconds=state['elapsed'];initial_metric=state['initial_metric']
 all_pred=dict(np.load(run/'predicciones_avance.npz'))
for epoch in range(first_epoch,10):
 torch.manual_seed(config['seed']+epoch);torch.cuda.manual_seed_all(config['seed']+epoch)
 frozen=epoch<=3
 for p in model.duration.parameters():p.requires_grad_(not frozen)
 model.train()
 if frozen:model.duration.eval()
 order=ids[torch.randperm(len(ids),device='cuda',generator=torch.Generator(device='cuda').manual_seed(config['seed']+1000+epoch))]
 total=0.;count=0
 for i in order.split(512):
  opt.zero_grad(set_to_none=True);loss=objective(i);assert torch.isfinite(loss);loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step();total+=float(loss.detach())*len(i);count+=len(i)
 pred=predict();all_pred['epoch_'+str(epoch)]=pred;m,preserve=retain(epoch,pred)
 row=dict(epoch=epoch,base_frozen=frozen,primary_train_n=count,train_loss=total/count,validation=m,preserved=preserve,seconds=previous_seconds+time.time()-start)
 history.append(row);write(run/'historia.json',history);print(json.dumps(row),flush=True)
 np.savez_compressed(run/'predicciones_avance.npz',**all_pred)
 torch.save(dict(state=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,history=history,saved=saved,best=best,best_epoch=best_epoch,best_file=best_file,
  initial_metric=initial_metric,elapsed=previous_seconds+time.time()-start),run/'ultima_reanudacion.pt')
best_pred=all_pred['protected_reference'] if best_epoch is None else all_pred['epoch_'+str(best_epoch)]
np.savez_compressed(run/'validacion.npz',row=va['row'],timestamp=va['timestamp'],true=v,pred=best_pred,**all_pred)
result=dict(id=run.name,config=config,validation=metric(v,best_pred),initial_validation=initial_metric,best_epoch=best_epoch,
 new_function_selected=best_epoch is not None,learned_update_selected=best_epoch is not None and best_epoch>0,
 selected_original_reference=best_epoch is None,all_epoch_validation=history,initial_model_state_sha256=initial_state_sha,
 model_file=best_file,model_sha256=sha(best_file),saved_checkpoints=saved,paired_vs_reference=paired(v,best_pred,ref['pred'],va['timestamp']),
 predictions_sha256=sha(run/'validacion.npz'),sources={s:sha(run/'fuentes'/s) for s in sources},Test_access=False,actual_data_files_accessed=sorted(access),
 parent_unchanged=sha(BEST/'modelo.pt')==config['parent_sha256'],seconds=previous_seconds+time.time()-start,goal='ACTIVE unless confirmation target later actually achieved')
assert result['parent_unchanged'];write(run/'resultado.json',result)
for name in ['ultima_reanudacion.pt','predicciones_avance.npz']:
 path=run/name
 if path.exists():assert path.resolve().is_relative_to(OUT.resolve());path.unlink()
print(json.dumps(dict(stage='complete',id=run.name,validation=result['validation'],best_epoch=best_epoch,saved=len(saved),seconds=result['seconds'])),flush=True)
