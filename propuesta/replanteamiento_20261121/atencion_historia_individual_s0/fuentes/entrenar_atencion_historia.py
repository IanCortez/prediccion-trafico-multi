"""Frozen-parent relational attention over64crossfit historical trips; unchanged primary Porto OD."""
from base_replanteo import *
import torch,argparse,shutil,gc
from inferencia_historica_fresca import load_frozen_history
from inferencia import feature_input
ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['individual','compressed'],default='individual');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
access=deny_test();start=time.time();proof=read(OUT/'verificacion_atencion_historia_cpu.json')
assert proof['passed'] and proof['model_sha256']==sha(OUT/'modelo_atencion_historia.py') and proof['token_constructor_sha256']==sha(OUT/'historia_individual.py')
torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction(.5)
torch.manual_seed(20261122);torch.cuda.manual_seed_all(20261122)
# Bind the historical parent's dependencies before loading the new token helper.
parent,base_ck,reference,reference_run,oldlib=load_frozen_history('historica_fresca_s0',device='cpu');del oldlib
from historia_individual import IndividualHistoryBank
from modelo_atencion_historia import IndividualHistoryAttention
from modelo_aux_subviajes import duration_score
assert reference['model_sha256']==proof['parent_sha256']
data=OUT/'datos_historia_individual_s0';meta=read(data/'resultado.json');normal=read(data/'normalizacion_Train_resultado.json')
assert normal['normalization_sha256']==proof['normalization_sha256']
constructor=dict(np.load(data/'normalizacion_Train.npz'));library=IndividualHistoryBank(Path(meta['shared_memory_file']),meta['memory_sha256'])
config=dict(mode=args.mode,seed=20261122,max_epochs=8,max_training_seconds=1800,batch=512,lr=.001,weight_decay=.0003,clip=5,
 parent_id=reference['id'],parent_model_file=reference['model_file'],parent_sha256=reference['model_sha256'],parent_validation=reference['validation'],
 parent_training='Frozen throughout; branch parameters only',architecture='Parent22components plus8query-conditioned slots over64source trips; canonical source ordering;4head64Dset attention.',
 loss='PrimaryNLL+5RPS+.2conditionalTraingeometry',precision='FP32 highest; canonicalFloat16derivedraw tokens',canonical_inference_batch=2048,candidate_head_eval_chunk=256,
 GPU_memory_fraction=.5,GPU_resource_note='At most oneGPU; increased internal cap from4.8to6GiB for individual-context inference, no storage or protocol change.',
 memory_sha256=meta['memory_sha256'],normalization_sha256=normal['normalization_sha256'],index_manifest_sha256=sha(data/'resultado.json'),
 source_query_inputs=['o_lon','o_lat','d_lon','d_lat','timestamp'],Train_n=1023485,Validation_n=127941,Test='sealed',
 control='Same network/initialweights/indices/batches, sourceG/path/time replaced by6prototype summaries; source relation fields unchanged.',
 control_trigger='At least.1s fullValidation improvement and upper exploratorydaybootstrap deltaCI<0, after frozen checkpoint certification.',
 preservation='Every complete initial/epoch metric and point prediction; every candidateMAE<131.06 and strict new globalbest; branch-only weights reference protectedparent.')
run=OUT/('atencion_historia_individual_s0' if args.mode=='individual' else 'control_compresion_historia_s0')
sources=['entrenar_atencion_historia.py','modelo_atencion_historia.py','historia_individual.py','inferencia_historica_fresca.py','base_replanteo.py','modelo_aux_subviajes.py']
if not args.resume:
 assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
 for name in sources:shutil.copy2(OUT/name,run/'fuentes'/name)
 shutil.copy2(data/'normalizacion_Train.npz',run/'constructor_tokens.npz');write(run/'protocolo.json',config)
 write(run/'referencia_fuentes_protegidas.json',dict(id=reference['id'],sources=reference['sources'],model_sha256=reference['model_sha256']))
else:
 assert read(run/'protocolo.json')==config
 for name in sources:assert sha(OUT/name)==sha(run/'fuentes'/name)
model=IndividualHistoryAttention(parent,constructor['mean'],constructor['std'],constructor['time_anchor'],args.mode)
initial_sha=hashlib.sha256()
for name,value in sorted(model.state_dict().items()):
 if not name.startswith('parent.'):initial_sha.update(name.encode());initial_sha.update(value.numpy().tobytes())
initial_sha=initial_sha.hexdigest()
if args.mode=='compressed':
 treatment=read(OUT/'atencion_historia_individual_s0/resultado.json');assert treatment['control_trigger_met']
 assert initial_sha==treatment['initial_branch_sha256']
model=model.cuda()
assert shutil.disk_usage(ROOT).free>3*1024**3
tr,X,names=feature_input('train',read(BEST/'resultado.json'));va,V,vnames=feature_input('validation',read(BEST/'resultado.json'))
assert names==vnames==base_ck['features'];x=torch.tensor(X,device='cuda');vx=torch.tensor(V,device='cuda');del X,V
public=[];summaries=[];queries=[]
for split,d in [('train',tr),('validation',va)]:
 cb=dict(np.load(OUT/'banco_proyeccion_incierta_s0'/(split+'_banco.npz')))
 hb=dict(np.load(OUT/'datos_prototipos_historicos_s0'/(split+'_historicos.npz')))
 file=data/(split+'_indices.npz');assert sha(file)==meta['outputs'][file.name]['sha256'];q=dict(np.load(file))
 for key in ['row','timestamp']:
  assert np.array_equal(d[key],cb[key]) and np.array_equal(d[key],hb[key]) and np.array_equal(d[key],q[key])
 public.append(torch.tensor(cb['bank'],device='cuda'));summaries.append(torch.tensor(hb['history'],device='cuda'))
 queries.append({key:torch.as_tensor(q[key],device='cuda') for key in ['xy_km','hour','weekend','indices']})
 del cb,hb,q
c,vc=public;h,vh=summaries;q,vq=queries;del public,summaries,queries
y=np.load(DATA/'train_y.npy');truth=np.load(DATA/'validation_y.npy');target=torch.tensor(y,device='cuda')
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index'];assert len(eligible)==1023485
assert np.max(tr['timestamp'][eligible]+y[eligible])<int(va['timestamp'].min());ids=torch.tensor(eligible,device='cuda')
gt=dict(np.load(DATA/'geometria_train.npz'));pos=np.searchsorted(gt['row'],tr['row']);safe=np.minimum(pos,len(gt['row'])-1)
valid=(pos<len(gt['row']))&(gt['row'][safe]==tr['row'])&(gt['X'][safe,-1]<=500)
geometry=gt['X'][safe,1:15].copy();valid&=np.isfinite(geometry).all(1);geometry[~valid]=0
g=torch.tensor(geometry,device='cuda');valid=torch.tensor(valid,device='cuda',dtype=torch.float32);del gt,geometry,pos,safe
reference_saved=dict(np.load(reference_run/'validacion.npz'))
for key in ['row','timestamp']:assert np.array_equal(reference_saved[key],va[key])
assert np.array_equal(reference_saved['true'],truth)
def raw_for(query,index):
 return library.features(query['xy_km'][index],query['hour'][index],query['weekend'][index],query['indices'][index])
def objective(index):
 raw=raw_for(q,index);out=model(x[index],c[index],h[index],raw);t=target[index]
 return duration_score(out[0],t,model.duration.duration_grid)+.2*model.duration.geometry_score(out,t,g[index],valid[index])
@torch.no_grad()
def predict(parent_only=False):
 gc.collect();torch.cuda.empty_cache();model.eval();parts=[]
 for first in range(0,len(vx),2048):
  end=min(first+2048,len(vx));n=end-first
  batch=[z[first:end] for z in [vx,vc,vh]]
  query={key:value[first:end] for key,value in vq.items()}
  if n<2048:
   batch=[torch.cat([z,z[-1:].expand((2048-n,)+z.shape[1:])]) for z in batch]
   query={key:torch.cat([z,z[-1:].expand((2048-n,)+z.shape[1:])]) for key,z in query.items()}
  if parent_only:out=model.parent(*batch)
  else:
   raw=library.features(query['xy_km'],query['hour'],query['weekend'],query['indices'])
   out=model(*batch,raw);del raw
  parts.append(model.duration.quantiles(out[0])[:n,1].cpu().numpy());del out
 return np.concatenate(parts)
# A realTrain full objective gradient preflight, before primary candidate evaluation.
model.train();preflight=objective(ids[:16]);assert torch.isfinite(preflight);preflight.backward()
assert all(p.grad is None for p in model.parent.parameters())
assert all(torch.isfinite(p.grad).all() for p in model.branch_parameters() if p.grad is not None)
model.zero_grad(set_to_none=True)
write(run/'verificacion_GPU_inicial.json',dict(passed=True,Train_loss=float(preflight.detach()),parent_gradients_absent=True,branch_parameters=sum(p.numel() for p in model.branch_parameters()),initial_branch_sha256=initial_sha))
parent_pred=predict(True);assert np.array_equal(parent_pred,reference_saved['pred'])
write(run/'verificacion_referencia.json',dict(passed=True,full127941point_predictions_exact=True,parent_sha256=reference['model_sha256'],Test=False))
opt=torch.optim.AdamW(model.branch_parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
history=[];saved=[];frontier=reference['validation']['mae_s'];best=float('inf');best_epoch=None;best_file=None;all_pred={'reference':parent_pred};elapsed_before=0;first_epoch=1;finished=False
def retain(epoch,pred):
 global best,best_epoch,best_file,frontier
 m=metric(truth,pred);keep=m['mae_s']<131.06 or m['mae_s']<frontier;file=None
 if keep:
  file=run/('epoca_'+str(epoch)+'.pt')
  branch={name:value for name,value in model.state_dict().items() if not name.startswith('parent.')}
  torch.save(dict(branch_state=branch,constructor=constructor,config=config,features=names,epoch=epoch,initial_branch_sha256=initial_sha),file)
  saved.append(dict(epoch=epoch,file=file.name,sha256=sha(file),validation=m,threshold_qualifies=m['mae_s']<131.06))
 if m['mae_s']<best:best=m['mae_s'];best_epoch=epoch;best_file=str(file) if file else None
 frontier=min(frontier,m['mae_s'])
 write(run/'candidatos_conservados.json',dict(candidates=saved,parent_reference_protected=True,Test='sealed'))
 return m,keep
if args.resume:
 s=torch.load(run/'ultima_reanudacion.pt',map_location='cpu',weights_only=False)
 check=model.load_state_dict(s['branch_state'],strict=False);assert not check.unexpected_keys and all(k.startswith('parent.') for k in check.missing_keys)
 opt.load_state_dict(s['optimizer']);history=s['history'];saved=s['saved'];frontier=s['frontier'];best=s['best'];best_epoch=s['best_epoch'];best_file=s['best_file']
 elapsed_before=s['elapsed'];first_epoch=s['epoch']+1;finished=s['finished'];initial=s['initial'];all_pred=dict(np.load(run/'predicciones_avance.npz'))
else:
 initial_pred=predict();all_pred['epoch_0']=initial_pred;initial,_=retain(0,initial_pred)
 print(json.dumps(dict(stage='initial_candidate',validation=initial,parent=reference['validation'],branch_parameters=sum(p.numel() for p in model.branch_parameters()))),flush=True)
for epoch in range(first_epoch,9):
 if finished:break
 model.train();torch.manual_seed(1000+epoch);torch.cuda.manual_seed_all(1000+epoch)
 order=ids[torch.randperm(len(ids),device='cuda',generator=torch.Generator(device='cuda').manual_seed(1000+epoch))]
 total=0.;count=0
 for index in order.split(512):
  opt.zero_grad(set_to_none=True);loss=objective(index);assert torch.isfinite(loss);loss.backward()
  torch.nn.utils.clip_grad_norm_(model.branch_parameters(),5,error_if_nonfinite=True);opt.step();total+=float(loss.detach())*len(index);count+=len(index)
 pred=predict();all_pred['epoch_'+str(epoch)]=pred;m,keep=retain(epoch,pred);elapsed=elapsed_before+time.time()-start
 finished=epoch>=8 or elapsed>=config['max_training_seconds']
 row=dict(epoch=epoch,primary_train_n=count,train_loss=total/count,validation=m,preserved=keep,seconds=elapsed);history.append(row);write(run/'historia.json',history);print(json.dumps(row),flush=True)
 np.savez_compressed(run/'predicciones_avance.npz',**all_pred)
 branch={name:value for name,value in model.state_dict().items() if not name.startswith('parent.')}
 torch.save(dict(branch_state=branch,optimizer=opt.state_dict(),epoch=epoch,history=history,saved=saved,frontier=frontier,best=best,best_epoch=best_epoch,best_file=best_file,elapsed=elapsed,finished=finished,initial=initial),run/'ultima_reanudacion.pt')
selected=all_pred['epoch_'+str(best_epoch)]
np.savez_compressed(run/'validacion.npz',row=va['row'],timestamp=va['timestamp'],true=truth,pred=selected,**all_pred)
comparison=paired(truth,selected,reference_saved['pred'],va['timestamp']);trigger=bool(comparison['delta_mae_s']<=-.1 and comparison['ci95_s'][1]<0)
result=dict(id=run.name,config=config,validation=metric(truth,selected),initial_validation=initial,best_epoch=best_epoch,epochs_run=len(history),all_epoch_validation=history,
 learned_update_selected=best_epoch>0,beats_parent=best<reference['validation']['mae_s'],control_trigger_met=trigger,paired_vs_parent=comparison,
 initial_branch_sha256=initial_sha,model_file=best_file,model_sha256=sha(best_file) if best_file else None,saved_checkpoints=saved,
 parent_unchanged=sha(reference['model_file'])==reference['model_sha256'],predictions_sha256=sha(run/'validacion.npz'),
 sources={name:sha(run/'fuentes'/name) for name in sources},Test_access=False,actual_data_files_accessed=sorted(access),seconds=elapsed_before+time.time()-start,
 goal='ACTIVE; no confirmation or external matchedcohort claim.')
assert result['parent_unchanged'];write(run/'resultado.json',result)
for name in ['ultima_reanudacion.pt','predicciones_avance.npz']:
 path=run/name
 if path.exists():assert path.resolve().is_relative_to(OUT.resolve());path.unlink()
print(json.dumps(dict(stage='complete',id=run.name,validation=result['validation'],epochs=len(history),best_epoch=best_epoch,beats_parent=result['beats_parent'],control_trigger_met=trigger,saved=len(saved),seconds=result['seconds'])),flush=True)
