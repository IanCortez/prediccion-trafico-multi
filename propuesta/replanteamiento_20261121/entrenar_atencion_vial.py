"""Fixed3frozen+6joint public-road composition trial, one GPU."""
from base_replanteo import *
import torch,argparse,shutil
from inferencia import feature_input
from modelo_mezcla_rutas import make_duration_model
from modelo_aux_subviajes import duration_score
from modelo_atencion_vial import PublicRouteAttention
ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['route','clock_control'],default='route');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
access=deny_test();data=OUT/'secuencias_viales_od_s0';ds=read(data/'resultado.json');gate=read(data/'verificacion_entrada.json')
assert gate['passed'] and ds['all_primary_rows_retained'] and not ds['Validation_labels_accessed'] and not ds['Test_access']
run=OUT/('atencion_vial_s0' if args.mode=='route' else 'control_reloj_atencion_vial_s0')
config=dict(mode=args.mode,seed=20261122,epochs=9,frozen_global_epochs=3,joint_epochs=6,batch=512,
 global_lr=1e-5,branch_lr=.001,weight_decay=.0003,geometry_weight=.2,rps_weight=5,clip=5,
 representation='32ordered public-road samples per length/freeflow route; learned16dim road ids,6static features,8calendar fields; shared64dim2layer4head Transformer and24time-parameter corrections. No observed query route.',
 information_control='Conditional separate run with all road tokens zero, same clock/context/architecture/loop. Tests road information, not a claim of equal effective capacity.',
 confidence='Train-only endpoint-independent sample counts n/(n+20) on road id embeddings; static map features available for all roads.',
 initialization='Protected best global mixture; correction head initially zero. First3epochs freeze global tower, then6joint epochs.',
 precision='FP32 highest, standard seeded CUDA, no bitwise training guarantee',canonical_inference_batch=2048,
 query=['o_lon','o_lat','d_lon','d_lat','timestamp'],primary_Train_n=1023485,Validation_n=127941,Test='sealed',
 parent_sha256=sha(BEST/'modelo.pt'),source_data_manifest_sha256=sha(data/'resultado.json'),
 preservation='Every completed metric and all epoch predictions, even MAE>131.06; weights for every strict primary new best and every complete MAE<131.06.')
sources=['entrenar_atencion_vial.py','modelo_atencion_vial.py','modelo_aux_subviajes.py','secuencias_viales_od.py','consulta_historicos_cruzados.py','base_replanteo.py']
if not args.resume:
 assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
 for name in sources:shutil.copy2(OUT/name,run/'fuentes'/name)
 write(run/'protocolo.json',config);write(run/'fuentes/parent_sources.json',{str(p.relative_to(FROZEN)):sha(p) for p in FROZEN.rglob('*.py')})
else:
 assert read(run/'protocolo.json')==config
 for name in sources:assert sha(OUT/name)==sha(run/'fuentes'/name)
start=time.time();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(False)
torch.cuda.set_per_process_memory_fraction(.4);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
rec=read(BEST/'resultado.json');ck=torch.load(BEST/'modelo.pt',map_location='cpu',weights_only=False)
duration=make_duration_model(ck,rec['config']);duration.load_state_dict(ck['state'])
tr,X,names=feature_input('train',rec);va,V,vnames=feature_input('validation',rec);assert names==vnames==rec['features']
train_tokens=dict(np.load(data/'train_secuencias.npz'));val_tokens=dict(np.load(data/'validation_secuencias.npz'))
for r,z in [(tr,train_tokens),(va,val_tokens)]:
 assert np.array_equal(r['row'],z['row']) and np.array_equal(r['timestamp'],z['timestamp'])
for name in ['train_secuencias.npz','validation_secuencias.npz']:assert sha(data/name)==ds['outputs'][name]['sha256']
raw=dict(np.load(DATA/'mapa_dirigido_2014.npz'));E=len(raw['u']);mid=(raw['xy'][raw['u']]+raw['xy'][raw['v']])/2/1000
direction=raw['xy'][raw['v']]-raw['xy'][raw['u']];direction/=np.maximum(np.linalg.norm(direction,axis=1)[:,None],1e-7)
static=np.vstack([np.zeros((1,6)),np.column_stack([mid,direction,np.log1p(raw['length']/100),np.log1p(raw['seconds']/30)])]).astype(np.float32)
counts=np.load(data/'frecuencias_Train.npy');assert len(counts)==E+1 and counts[0]==0
confidence=counts.astype(np.float32)/(counts.astype(np.float32)+20);clock=[names.index(n) for n in ['hour_sin','hour_cos','week_sin','week_cos','season_sin','season_cos','weekend','calendar_special']]
model=PublicRouteAttention(duration,static,confidence,clock,args.mode).cuda()
x=torch.tensor(X,device='cuda');vx=torch.tensor(V,device='cuda')
tokens=torch.tensor(train_tokens['tokens'],device='cuda',dtype=torch.long);vtokens=torch.tensor(val_tokens['tokens'],device='cuda',dtype=torch.long)
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index'];ids=torch.tensor(eligible,device='cuda');assert len(eligible)==1023485
y=np.load(DATA/'train_y.npy');v=np.load(DATA/'validation_y.npy');target=torch.tensor(y,device='cuda')
gt=dict(np.load(DATA/'geometria_train.npz'));pos=np.searchsorted(gt['row'],tr['row']);safe=np.minimum(pos,len(gt['row'])-1)
valid=(pos<len(gt['row']))&(gt['row'][safe]==tr['row'])&(gt['X'][safe,-1]<=500)
geometry=gt['X'][safe,1:15].copy();valid&=np.isfinite(geometry).all(1);geometry[~valid]=0
g=torch.tensor(geometry,device='cuda');mask=torch.tensor(valid,device='cuda',dtype=torch.float32)
del X,V,train_tokens,val_tokens,raw,gt,geometry
ref=dict(np.load(BEST/'validacion.npz'));assert np.array_equal(ref['row'],va['row']) and np.array_equal(ref['true'],v)
@torch.no_grad()
def predict():
 model.eval();parts=[]
 for first in range(0,len(vx),2048):
  out=model(vx[first:first+2048],vtokens[first:first+2048]);parts.append(model.duration.quantiles(out[0])[:,1].cpu().numpy())
 return np.concatenate(parts)
def objective(index):
 out=model(x[index],tokens[index]);t=target[index]
 return duration_score(out[0],t,model.duration.duration_grid)+.2*model.duration.geometry_score(out,t,g[index],mask[index])
opt=torch.optim.AdamW([dict(params=model.duration.parameters(),lr=1e-5),dict(params=model.branch_parameters(),lr=.001)],weight_decay=.0003)
history=[];saved=[];all_pred={'epoch_0':ref['pred']};best=rec['validation']['mae_s'];best_epoch=0;best_file=str(BEST/'modelo.pt');first_epoch=1;previous_seconds=0.;road_gradient_checked=False
if not args.resume:
 model.eval();assert np.array_equal(predict(),ref['pred'])
 opt.zero_grad(set_to_none=True);loss=objective(ids[:32]);assert torch.isfinite(loss);loss.backward()
 assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 assert float(model.readout[-1].weight.grad.norm())>0;opt.zero_grad(set_to_none=True)
 write(run/'verificacion_GPU_inicial.json',dict(passed=True,zero_correction_all127941_reference_predictions_exact=True,finite_gradients=True,correction_output_gradient_nonzero=True,Test_access=False))
 print(json.dumps(dict(stage='road_attention_GPU_gate',passed=True,reference_exact=True,parameters=sum(p.numel() for p in model.parameters()))),flush=True)
else:
 state=torch.load(run/'ultima_reanudacion.pt',map_location='cuda',weights_only=False);model.load_state_dict(state['state']);opt.load_state_dict(state['optimizer'])
 history=state['history'];saved=state['saved'];best=state['best'];best_epoch=state['best_epoch'];best_file=state['best_file'];first_epoch=state['epoch']+1;previous_seconds=state['elapsed'];road_gradient_checked=state['road_gradient_checked']
 all_pred=dict(np.load(run/'predicciones_avance.npz'))
for epoch in range(first_epoch,10):
 torch.manual_seed(config['seed']+epoch);torch.cuda.manual_seed_all(config['seed']+epoch)
 frozen=epoch<=3
 for p in model.duration.parameters():p.requires_grad_(not frozen)
 model.train()
 if frozen:model.duration.eval()
 order=ids[torch.randperm(len(ids),device='cuda',generator=torch.Generator(device='cuda').manual_seed(config['seed']+epoch+1000))]
 total=0.;count=0
 for step,index in enumerate(order.split(512)):
  opt.zero_grad(set_to_none=True);loss=objective(index);assert torch.isfinite(loss);loss.backward()
  if not road_gradient_checked and step==2:
   grad=float(model.edge_embedding.weight.grad.norm())
   assert args.mode=='clock_control' or grad>0
   write(run/'verificacion_gradiente_vial.json',dict(passed=True,edge_embedding_gradient_norm=grad,global_parameters_frozen=frozen,only_Train_queries=True))
   road_gradient_checked=True
  torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
  total+=float(loss.detach())*len(index);count+=len(index)
 pred=predict();all_pred['epoch_'+str(epoch)]=pred;m=metric(v,pred);improved=m['mae_s']<best;preserve=improved or m['mae_s']<131.06
 path=run/('epoca_'+str(epoch)+'.pt')
 if preserve:
  torch.save(dict(state=model.state_dict(),config=config,parent_config=rec['config'],
   original_constructor={k:ck[k] for k in ['mean','std','keep','bins','anchors','anchor_indices','geometry_indices']},features=names,epoch=epoch),path)
  saved.append(dict(epoch=epoch,file=path.name,sha256=sha(path),validation=m,threshold_qualifies=m['mae_s']<131.06))
 if improved:best=m['mae_s'];best_epoch=epoch;best_file=str(path)
 row=dict(epoch=epoch,global_frozen=frozen,primary_train_n=count,train_loss=total/count,validation=m,preserved=preserve,seconds=previous_seconds+time.time()-start)
 history.append(row);write(run/'historia.json',history);print(json.dumps(row),flush=True)
 np.savez_compressed(run/'predicciones_avance.npz',**all_pred)
 torch.save(dict(state=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,history=history,saved=saved,best=best,best_epoch=best_epoch,best_file=best_file,road_gradient_checked=road_gradient_checked,elapsed=previous_seconds+time.time()-start),run/'ultima_reanudacion.pt')
best_pred=all_pred['epoch_'+str(best_epoch)]
np.savez_compressed(run/'validacion.npz',row=va['row'],timestamp=va['timestamp'],true=v,pred=best_pred,**all_pred)
result=dict(id=run.name,config=config,validation=metric(v,best_pred),best_epoch=best_epoch,learned_update_selected=best_epoch>0,all_epoch_validation=history,
 model_file=best_file,model_sha256=sha(best_file),saved_checkpoints=saved,paired_vs_reference=paired(v,best_pred,ref['pred'],va['timestamp']),
 predictions_sha256=sha(run/'validacion.npz'),sources={s:sha(run/'fuentes'/s) for s in sources},Test_access=False,actual_data_files_accessed=sorted(access),
 parent_unchanged=sha(BEST/'modelo.pt')==config['parent_sha256'],seconds=previous_seconds+time.time()-start,goal='ACTIVE unless confirmation target actually later achieved')
assert result['parent_unchanged'];write(run/'resultado.json',result)
for name in ['ultima_reanudacion.pt','predicciones_avance.npz']:
 path=run/name
 if path.exists():assert path.resolve().is_relative_to(OUT.resolve());path.unlink()
print(json.dumps(dict(stage='complete',id=run.name,validation=result['validation'],best_epoch=best_epoch,saved=len(saved),seconds=result['seconds'])),flush=True)
