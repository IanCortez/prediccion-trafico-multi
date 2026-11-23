"""Fresh22component model with crossfit historical route/time prototypes."""
from base_replanteo import *
import torch,argparse,shutil
from inferencia import feature_input
from modelo_mezcla_rutas import make_duration_model
from modelo_mezcla_proyeccion import ProjectionMixture
from modelo_mezcla_historica import HistoricalRouteMixture
from modelo_aux_subviajes import duration_score
ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['historical','parent_control'],default='historical');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
assert args.mode=='parent_control'
access=deny_test();proof=read(OUT/'verificacion_mezcla_historica_fresca.json');assert proof['passed']
assert proof['model_source_sha256']==sha(OUT/'modelo_mezcla_historica.py') and proof['parent_model_source_sha256']==sha(OUT/'modelo_mezcla_proyeccion.py')
data=OUT/'datos_prototipos_historicos_s0';meta=read(data/'resultado.json');fix=read(data/'calibracion_fp64.json');mapdata=OUT/'banco_proyeccion_incierta_s0';mapmeta=read(mapdata/'resultado.json')
assert meta['own_fold_intervention_passed'] and fix['rank']==2 and proof['history_constructor_sha256']==fix['used_constructor_sha256']
reference=read(OUT/'historica_fresca_s0/referencia_previa.json')
run=OUT/('historica_fresca_s0' if args.mode=='historical' else 'control_historica_fresca_s0')
config=dict(mode=args.mode,seed=0,max_epochs=50,min_epochs=20,patience=12,meaningful_progress_s=.01,max_training_seconds=3600,
 batch=512,lr=.001,weight_decay=.0003,clip=5,geometry_weight=.2,rps_weight=5,
 scheduler='ReduceLROnPlateau factor.5 patience5 min1e-5',initial_historical_gate_logit=0,
 architecture='Fresh global8unknown-route modes,8public-projection modes,6crossfit historical route/time modes; shared heads and learned set-conditioned gates.',
 initialization='All neural weights trained from scratch; no retained model state loaded. Reuse only eligibleTrain input normalization/bins/physical anchors.',
 historical_control='Same model/state, preprocessing,primary examples and optimizer; historical branch unavailable. An ablation of the branch, not equal effective capacity.',
 query=['o_lon','o_lat','d_lon','d_lat','timestamp'],Train_n=1023485,Validation_n=127941,Test='sealed',
 canonical_inference_batch=2048,inference_candidate_chunk=256,precision='FP32 highest model,canonicalFP16derived route/history inputs,standard seededCUDA',
 base_constructor_sha256=proof['base_constructor_sha256'],history_constructor_sha256=fix['used_constructor_sha256'],
 map_normalization_sha256=mapmeta['normalization_sha256'],history_data_manifest_sha256=sha(data/'resultado.json'),
 preservation='All metrics and every epoch prediction. Checkpoint every strict global new best and every completeMAE<131.06.',
 reference_mae_s=reference['validation']['mae_s'],reference_checkpoint_sha256=reference['model_sha256'])
sources=['entrenar_historica_fresca_control.py','modelo_mezcla_historica.py','modelo_mezcla_proyeccion.py','modelo_aux_subviajes.py','historicos_rutas.py','piloto_prototipos_ruta.py','rutas_proyeccion_incierta.py','consulta_historicos_cruzados.py','base_replanteo.py']
if not args.resume:
 assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
 for s in sources:shutil.copy2(OUT/s,run/'fuentes'/s)
 shutil.copy2(OUT/'campo_multiescala_s0/constructor.pt',run/'constructor_base.pt')
 shutil.copy2(data/fix['used_constructor_file'],run/'constructor_historia.npz')
 shutil.copy2(mapdata/'normalizacion_Train.npz',run/'constructor_mapa.npz')
 write(run/'protocolo.json',config);write(run/'fuentes/original_sources.json',{str(p.relative_to(FROZEN)):sha(p) for p in FROZEN.rglob('*.py')})
else:
 assert read(run/'protocolo.json')==config
 for s in sources:assert sha(OUT/s)==sha(run/'fuentes'/s)
start=time.time();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(False);torch.cuda.set_per_process_memory_fraction(.4)
assert shutil.disk_usage(ROOT).free>3*1024**3
base_ctor=torch.load(run/'constructor_base.pt',map_location='cpu',weights_only=False)
assert 'state' not in base_ctor and base_ctor['Train_n']==1023485 and sha(run/'constructor_base.pt')==config['base_constructor_sha256']
hctor=dict(np.load(run/'constructor_historia.npz'));mctor=dict(np.load(run/'constructor_mapa.npz'))
assert sha(run/'constructor_historia.npz')==config['history_constructor_sha256'] and sha(run/'constructor_mapa.npz')==config['map_normalization_sha256']
rec=read(BEST/'resultado.json');assert base_ctor['features']==rec['features']
torch.manual_seed(0);torch.cuda.manual_seed_all(0)
duration=make_duration_model(base_ctor,rec['config']);parent=ProjectionMixture(duration,mctor['mean'],mctor['std'],'expanded')
model=HistoricalRouteMixture(parent,hctor['mean'],hctor['std'],hctor['time_anchor'],args.mode)
with torch.no_grad():model.gate[-1].bias.zero_()
h=hashlib.sha256()
for name,p in sorted(model.state_dict().items()):h.update(name.encode());h.update(p.detach().numpy().tobytes())
initial_sha=h.hexdigest()
if args.mode=='parent_control':
 treatment=read(OUT/'historica_fresca_s0/resultado.json');assert initial_sha==treatment['initial_model_state_sha256']
model=model.cuda()
tr,X,names=feature_input('train',rec);va,V,vnames=feature_input('validation',rec);assert names==vnames==rec['features']
route=[];history=[]
for split,d in [('train',tr),('validation',va)]:
 p=mapdata/(split+'_banco.npz');assert sha(p)==mapmeta['outputs'][p.name]['sha256'];z=dict(np.load(p))
 assert np.array_equal(z['row'],d['row']) and np.array_equal(z['timestamp'],d['timestamp']);route.append(torch.tensor(z['bank'],device='cuda',dtype=torch.float16));del z
 p=data/(split+'_historicos.npz');assert sha(p)==meta['outputs'][p.name]['sha256'];z=dict(np.load(p))
 assert np.array_equal(z['row'],d['row']) and np.array_equal(z['timestamp'],d['timestamp']);history.append(torch.tensor(z['history'],device='cuda',dtype=torch.float16));del z
c,vc=route;hinput,vh=history;del route,history
x=torch.tensor(X,device='cuda');vx=torch.tensor(V,device='cuda');y=np.load(DATA/'train_y.npy');v=np.load(DATA/'validation_y.npy');target=torch.tensor(y,device='cuda')
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index'];assert len(eligible)==1023485;ids=torch.tensor(eligible,device='cuda')
gt=dict(np.load(DATA/'geometria_train.npz'));pos=np.searchsorted(gt['row'],tr['row']);safe=np.minimum(pos,len(gt['row'])-1)
mask=(pos<len(gt['row']))&(gt['row'][safe]==tr['row'])&(gt['X'][safe,-1]<=500);geometry=gt['X'][safe,1:15].copy();mask&=np.isfinite(geometry).all(1);geometry[~mask]=0
g=torch.tensor(geometry,device='cuda');valid=torch.tensor(mask,device='cuda',dtype=torch.float32);del X,V,gt,geometry,mask
ref=dict(np.load(OUT/reference['id']/'validacion.npz'));assert np.array_equal(ref['row'],va['row']) and np.array_equal(ref['true'],v)
assert sha(reference['model_file'])==config['reference_checkpoint_sha256']
@torch.no_grad()
def predict():
 model.eval();parts=[]
 for first in range(0,len(vx),2048):
  out=model(vx[first:first+2048],vc[first:first+2048],vh[first:first+2048]);parts.append(model.duration.quantiles(out[0])[:,1].cpu().numpy())
 return np.concatenate(parts)
def objective(index):
 out=model(x[index],c[index],hinput[index]);t=target[index]
 return duration_score(out[0],t,model.duration.duration_grid)+.2*model.duration.geometry_score(out,t,g[index],valid[index])
opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0003)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=5,min_lr=1e-5)
all_pred={'reference':ref['pred']};history=[];saved=[];discarded=[];frontier=config['reference_mae_s'];best=float('inf');best_epoch=0;best_file=None;first_epoch=1;elapsed_before=0.;early_best=float('inf');bad=0;finished=False
def retain(epoch,pred):
 global best,best_epoch,best_file,frontier
 m=metric(v,pred);old_file=best_file;old_score=best;improves_run=m['mae_s']<best;save=improves_run or m['mae_s']<frontier or m['mae_s']<131.06;path=run/('epoca_'+str(epoch)+'.pt')
 if save:
  torch.save(dict(state=model.state_dict(),config=config,base_constructor=base_ctor,base_config=rec['config'],
   map_constructor=mctor,history_constructor=hctor,features=names,epoch=epoch),path)
  saved.append(dict(epoch=epoch,file=path.name,sha256=sha(path),validation=m,threshold_qualifies=m['mae_s']<131.06));frontier=min(frontier,m['mae_s'])
 if improves_run:
  best=m['mae_s'];best_epoch=epoch;best_file=str(path)
  if old_file and old_file!=str(path) and old_score>=131.06 and old_score>=config['reference_mae_s']:
   previous=Path(old_file).resolve();assert previous.parent==run.resolve() and previous.stat().st_size<64*1024**2
   discarded.append(dict(file=previous.name,sha256=sha(previous),mae_s=old_score,reason='Superseded nonqualifying control checkpoint; every metric/prediction retained.'))
   previous.unlink();saved[:]=[entry for entry in saved if entry['file']!=previous.name]
 return m,save
if not args.resume:
 model.eval();opt.zero_grad(set_to_none=True);loss=objective(ids[:32]);assert torch.isfinite(loss);loss.backward()
 assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 assert args.mode=='parent_control' or model.history_head[-1].weight.grad.norm()>0
 opt.zero_grad(set_to_none=True)
 with torch.no_grad():
  out=model(x[ids[:32]],c[ids[:32]],hinput[ids[:32]]);label=((target[ids[:32]]-60)/15).long();J=out[1].shape[-1]
  assert J==22
  lp=out[0].gather(-1,label[:,None,None].expand(-1,16,1))[...,0];cp=out[4].gather(-1,label[:,None,None,None].expand(-1,16,J,1))[...,0]
  error=float(((out[1]+cp-lp[...,None]).exp().sum(-1)-1).abs().max());assert error<1e-5
 write(run/'verificacion_GPU_inicial.json',dict(passed=True,fresh_neural_weights=True,no_retained_network_state_loaded=True,finite_joint_gradients=True,posterior_component_sum_error=error,Test_access=False))
 pred=predict();all_pred['epoch_0']=pred;initial,preserve=retain(0,pred);early_best=initial['mae_s']
 print(json.dumps(dict(stage='fresh_historical_GPU_gate',mode=args.mode,passed=True,initial_validation=initial,parameters=sum(p.numel() for p in model.parameters()),initial_state_sha256=initial_sha)),flush=True)
else:
 state=torch.load(run/'ultima_reanudacion.pt',map_location='cuda',weights_only=False);model.load_state_dict(state['state']);opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
 history=state['history'];saved=state['saved'];discarded=state.get('discarded',[]);frontier=state['frontier'];best=state['best'];best_epoch=state['best_epoch'];best_file=state['best_file'];first_epoch=state['epoch']+1;elapsed_before=state['elapsed'];early_best=state['early_best'];bad=state['bad'];finished=state['finished'];initial=state['initial']
 all_pred=dict(np.load(run/'predicciones_avance.npz'))
for epoch in range(first_epoch,51):
 if finished:break
 torch.manual_seed(epoch);torch.cuda.manual_seed_all(epoch);model.train()
 order=ids[torch.randperm(len(ids),device='cuda',generator=torch.Generator(device='cuda').manual_seed(1000+epoch))]
 total=0.;count=0
 for i in order.split(512):
  opt.zero_grad(set_to_none=True);loss=objective(i);assert torch.isfinite(loss);loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step();total+=float(loss.detach())*len(i);count+=len(i)
 pred=predict();all_pred['epoch_'+str(epoch)]=pred;m,preserved=retain(epoch,pred);scheduler.step(m['mae_s'])
 if m['mae_s']<early_best-.01:early_best=m['mae_s'];bad=0
 else:bad+=1
 elapsed=elapsed_before+time.time()-start;finished=epoch>=20 and bad>=12 or elapsed>=3600 or epoch>=50
 row=dict(epoch=epoch,primary_train_n=count,train_loss=total/count,validation=m,preserved=preserved,lr=opt.param_groups[0]['lr'],seconds=elapsed);history.append(row)
 write(run/'historia.json',history);print(json.dumps(row),flush=True);np.savez_compressed(run/'predicciones_avance.npz',**all_pred)
 torch.save(dict(state=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,history=history,saved=saved,discarded=discarded,frontier=frontier,best=best,best_epoch=best_epoch,best_file=best_file,
  elapsed=elapsed,early_best=early_best,bad=bad,finished=finished,initial=initial),run/'ultima_reanudacion.pt')
best_pred=all_pred['epoch_'+str(best_epoch)]
np.savez_compressed(run/'validacion.npz',row=va['row'],timestamp=va['timestamp'],true=v,pred=best_pred,**all_pred)
result=dict(id=run.name,config=config,validation=metric(v,best_pred),initial_validation=initial,best_epoch=best_epoch,epochs_run=len(history),all_epoch_validation=history,
 trained_from_scratch=True,initial_model_state_sha256=initial_sha,beats_verified_reference=best<config['reference_mae_s'],
 model_file=best_file,model_sha256=sha(best_file) if best_file else None,saved_checkpoints=saved,discarded_control_checkpoints=discarded,
 paired_vs_reference=paired(v,best_pred,ref['pred'],va['timestamp']),predictions_sha256=sha(run/'validacion.npz'),
 sources={s:sha(run/'fuentes'/s) for s in sources},Test_access=False,actual_data_files_accessed=sorted(access),seconds=elapsed_before+time.time()-start,
 reference_unchanged=sha(reference['model_file'])==config['reference_checkpoint_sha256'],goal='ACTIVE unless strong frozen model later achieves73.5confirmation.')
assert result['reference_unchanged'];write(run/'resultado.json',result)
for name in ['ultima_reanudacion.pt','predicciones_avance.npz']:
 path=run/name
 if path.exists():assert path.resolve().is_relative_to(OUT.resolve());path.unlink()
print(json.dumps(dict(stage='complete',id=run.name,validation=result['validation'],best_epoch=best_epoch,epochs=len(history),beats_reference=result['beats_verified_reference'],saved=len(saved),seconds=result['seconds'])),flush=True)
