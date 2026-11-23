"""Explicit FP32 point-MAE experiment on unchanged Porto OD; Test sealed."""
from base_replanteo import *
import argparse,torch,shutil,gc
from inferencia import feature_input
from inferencia_historica_fresca import load_frozen_history
from modelo_mediana_directa import DirectMedianReadout
ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['joint','frozen_control'],default='joint');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
access=deny_test();start=time.time();proof=read(OUT/'verificacion_mediana_directa_cpu.json')
assert proof['passed'] and proof['source_sha256']==sha(OUT/'modelo_mediana_directa.py')
torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction(.4)
torch.manual_seed(20261122);torch.cuda.manual_seed_all(20261122)
reference,base_ck,base_record,base_run,library=load_frozen_history('historica_fresca_s0',device='cpu');del library
assert base_record['model_sha256']==proof['reference_sha256']
assert shutil.disk_usage(ROOT).free>3*1024**3
config=dict(mode=args.mode,seed=20261122,max_epochs=6,warmup_epochs=2,batch=512,head_lr=.0003,student_lr=.00001,
 weight_decay=.0003,clip=5,max_training_seconds=1800,precision='FP32 highest',canonical_inference_batch=2048,
 loss='.5MAE(mean members)+.5mean memberMAE, divided by600; no implicit quantile gradients.',
 reference_id=base_record['id'],reference_model_file=base_record['model_file'],reference_sha256=base_record['model_sha256'],
 reference_validation=base_record['validation'],query=['o_lon','o_lat','d_lon','d_lat','timestamp'],
 Train_n=1023485,Validation_n=127941,Test='sealed',point_output='Mean of16direct member outputs, nearest original15second bin',
 control='Same fixed readout and batches; student representation remains frozen and eval throughout.',
 uncertainty='This is a point readout; frozen reference intervals are not certified intervals of the new point predictor.',
 preservation='Every epoch metrics and predictions; every sub131.06checkpoint and strict new best. Original and verified references untouched.')
run=OUT/('mediana_directa_s0' if args.mode=='joint' else 'control_mediana_directa_s0')
sources=['entrenar_mediana_directa.py','entrenar_mediana_directa_recuperar.py','modelo_mediana_directa.py','inferencia_historica_fresca.py','base_replanteo.py']
if not args.resume:
 assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
 for name in sources:shutil.copy2(OUT/name,run/'fuentes'/name)
 write(run/'protocolo.json',config);write(run/'referencia_fuentes_protegidas.json',dict(id=base_record['id'],sources=base_record['sources'],model_sha256=base_record['model_sha256']))
else:
 assert read(run/'protocolo.json')==config
 for name in sources:assert sha(OUT/name)==sha(run/'fuentes'/name)
model=DirectMedianReadout(reference).cuda().eval()
initial_hash=hashlib.sha256()
for name,value in sorted(model.state_dict().items()):
 if not name.startswith('reference.'):initial_hash.update(name.encode());initial_hash.update(value.detach().cpu().numpy().tobytes())
initial_hash=initial_hash.hexdigest()
if args.mode=='frozen_control':
 treated=read(OUT/'mediana_directa_s0/resultado.json');assert initial_hash==treated['initial_student_readout_sha256']
tr,X,names=feature_input('train',read(BEST/'resultado.json'));va,V,vnames=feature_input('validation',read(BEST/'resultado.json'))
assert names==vnames==base_ck['features'];x=torch.tensor(X,device='cuda');vx=torch.tensor(V,device='cuda');del X,V
banks=[];histories=[]
for split,d in [('train',tr),('validation',va)]:
 z=dict(np.load(OUT/'banco_proyeccion_incierta_s0'/(split+'_banco.npz')))
 h=dict(np.load(OUT/'datos_prototipos_historicos_s0'/(split+'_historicos.npz')))
 for key in ['row','timestamp']:assert np.array_equal(z[key],d[key]) and np.array_equal(h[key],d[key])
 banks.append(torch.tensor(z['bank'],device='cuda'));histories.append(torch.tensor(h['history'],device='cuda'));del z,h
c,vc=banks;h,vh=histories;del banks,histories
def padded(first,xx,cc,hh):
 batch=[z[first:first+2048] for z in [xx,cc,hh]];n=len(batch[0])
 if n<2048:batch=[torch.cat([z,z[-1:].expand((2048-n,)+z.shape[1:])]) for z in batch]
 return batch,n
@torch.no_grad()
def frozen_points(xx,cc,hh):
 model.eval();parts=[]
 for first in range(0,len(xx),2048):
  batch,n=padded(first,xx,cc,hh);parts.append(model.reference_point(*batch)[:n].cpu().numpy().astype(np.int16))
 return np.concatenate(parts)
cache=OUT/'datos_mediana_directa_s0'
cache_key=dict(reference_sha256=base_record['model_sha256'],feature_names=names,
 train_row_timestamp_sha256=hashlib.sha256(tr['row'].tobytes()+tr['timestamp'].tobytes()).hexdigest(),
 validation_row_timestamp_sha256=hashlib.sha256(va['row'].tobytes()+va['timestamp'].tobytes()).hexdigest(),
 public_train_sha256=sha(OUT/'banco_proyeccion_incierta_s0/train_banco.npz'),
 public_validation_sha256=sha(OUT/'banco_proyeccion_incierta_s0/validation_banco.npz'),
 history_train_sha256=sha(OUT/'datos_prototipos_historicos_s0/train_historicos.npz'),
 history_validation_sha256=sha(OUT/'datos_prototipos_historicos_s0/validation_historicos.npz'))
if not cache.exists():
 cache.mkdir();point_train=frozen_points(x,c,h);point_val=frozen_points(vx,vc,vh)
 np.savez_compressed(cache/'puntos_referencia.npz',train=point_train,validation=point_val)
 write(cache/'resultado.json',dict(key=cache_key,predictions_sha256=sha(cache/'puntos_referencia.npz'),bytes=(cache/'puntos_referencia.npz').stat().st_size,labels_used=False,Test=False))
else:
 cache_meta=read(cache/'resultado.json');assert cache_meta['key']==cache_key and sha(cache/'puntos_referencia.npz')==cache_meta['predictions_sha256']
 z=dict(np.load(cache/'puntos_referencia.npz'));point_train=z['train'];point_val=z['validation']
assert sum(q.stat().st_size for q in cache.iterdir() if q.is_file())<10*1024**2
base_saved=dict(np.load(base_run/'validacion.npz'))
assert np.array_equal(point_val,base_saved['pred'])
tpoint=torch.tensor(point_train,device='cuda',dtype=torch.float32);vpoint=torch.tensor(point_val,device='cuda',dtype=torch.float32)
y=np.load(DATA/'train_y.npy');truth=np.load(DATA/'validation_y.npy');target=torch.tensor(y,device='cuda')
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index'];assert len(eligible)==1023485
assert np.max(tr['timestamp'][eligible]+y[eligible])<int(va['timestamp'].min())
ids=torch.tensor(eligible,device='cuda')
@torch.no_grad()
def predict():
 gc.collect();torch.cuda.empty_cache();model.eval();continuous=[];points=[]
 for first in range(0,len(vx),2048):
  batch,n=padded(first,vx,vc,vh);rp=vpoint[first:first+n]
  if n<2048:rp=torch.cat([rp,rp[-1:].expand(2048-n)])
  members,student_distribution=model.compute(*batch,rp);del student_distribution;value=members.mean(1)
  continuous.append(value[:n].cpu().numpy());points.append(model.quantize(value)[:n].cpu().numpy())
 return np.concatenate(points),np.concatenate(continuous)
original,continuous=predict();assert np.array_equal(original,base_saved['pred'])
opt=torch.optim.AdamW([dict(params=list(model.readout.parameters()),lr=config['head_lr']),dict(params=list(model.student.parameters()),lr=config['student_lr'])],weight_decay=config['weight_decay'])
history=[];saved=[];all_predictions={'epoch_0':original};all_continuous={'epoch_0':continuous}
best=base_record['validation']['mae_s'];best_epoch=0;best_file=None;frontier=best;elapsed_before=0;first_epoch=1;finished=False
if args.resume:
 s=torch.load(run/'ultima_reanudacion.pt',map_location='cpu',weights_only=False)
 model.student.load_state_dict(s['student_state']);model.readout.load_state_dict(s['readout_state']);opt.load_state_dict(s['optimizer'])
 history=s['history'];saved=s['saved'];best=s['best'];best_epoch=s['best_epoch'];best_file=s['best_file'];frontier=s['frontier'];elapsed_before=s['elapsed'];first_epoch=s['epoch']+1;finished=s['finished']
 z=dict(np.load(run/'predicciones_avance.npz'));all_predictions={k:v for k,v in z.items() if k.startswith('epoch_')};all_continuous={k[11:]:v for k,v in z.items() if k.startswith('continuous_')}
for epoch in range(first_epoch,config['max_epochs']+1):
 if finished:break
 joint=args.mode=='joint' and epoch>2;model.set_student_training(joint);model.train()
 torch.manual_seed(1000+epoch);torch.cuda.manual_seed_all(1000+epoch)
 order=ids[torch.randperm(len(ids),device='cuda',generator=torch.Generator(device='cuda').manual_seed(1000+epoch))]
 total=0.;count=0
 for ix in order.split(512):
  opt.zero_grad(set_to_none=True);members,student_distribution=model.training_forward(x[ix],c[ix],h[ix],tpoint[ix]);del student_distribution
  loss=(.5*(members.mean(1)-target[ix]).abs().mean()+.5*(members-target[ix,None]).abs().mean())/600
  assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5,error_if_nonfinite=True);opt.step()
  total+=float(loss.detach())*len(ix);count+=len(ix)
 pred,cont=predict();m=metric(truth,pred);all_predictions['epoch_'+str(epoch)]=pred;all_continuous['epoch_'+str(epoch)]=cont
 keep=m['mae_s']<131.06 or m['mae_s']<frontier;file=None
 if keep:
  file=run/('epoca_'+str(epoch)+'.pt')
  torch.save(dict(student_state=model.student.state_dict(),readout_state=model.readout.state_dict(),reference_id=base_record['id'],reference_sha256=base_record['model_sha256'],
   config=config,epoch=epoch,features=names,initial_student_readout_sha256=initial_hash),file)
  saved.append(dict(epoch=epoch,file=file.name,sha256=sha(file),validation=m,threshold_qualifies=m['mae_s']<131.06))
 if m['mae_s']<best:best=m['mae_s'];best_epoch=epoch;best_file=str(file)
 frontier=min(frontier,m['mae_s'])
 elapsed=elapsed_before+time.time()-start;finished=epoch>=6 or elapsed>=config['max_training_seconds']
 row=dict(epoch=epoch,student_trainable=joint,primary_train_n=count,train_loss=total/count,validation=m,continuous_validation=metric(truth,cont),preserved=keep,seconds=elapsed)
 history.append(row);write(run/'historia.json',history);print(json.dumps(row),flush=True)
 np.savez_compressed(run/'predicciones_avance.npz',**all_predictions,**{'continuous_'+k:v for k,v in all_continuous.items()})
 torch.save(dict(student_state=model.student.state_dict(),readout_state=model.readout.state_dict(),optimizer=opt.state_dict(),epoch=epoch,history=history,saved=saved,
  best=best,best_epoch=best_epoch,best_file=best_file,frontier=frontier,elapsed=elapsed,finished=finished),run/'ultima_reanudacion.pt')
selected=all_predictions['epoch_'+str(best_epoch)]
np.savez_compressed(run/'validacion.npz',row=va['row'],timestamp=va['timestamp'],true=truth,pred=selected,
 continuous_pred=all_continuous['epoch_'+str(best_epoch)],**all_predictions,**{'continuous_'+k:v for k,v in all_continuous.items()})
result=dict(id=run.name,config=config,validation=metric(truth,selected),initial_validation=base_record['validation'],best_epoch=best_epoch,epochs_run=len(history),all_epoch_validation=history,
 initial_student_readout_sha256=initial_hash,learned_update_selected=best_epoch>0,
 model_file=best_file,model_sha256=sha(best_file) if best_file else None,saved_checkpoints=saved,
 reference_unchanged=sha(base_record['model_file'])==base_record['model_sha256'],
 paired_vs_reference=paired(truth,selected,base_saved['pred'],va['timestamp']),
 predictions_sha256=sha(run/'validacion.npz'),reference_cache_sha256=sha(cache/'puntos_referencia.npz'),
 sources={s:sha(run/'fuentes'/s) for s in sources},Test_access=False,actual_data_files_accessed=sorted(access),
 seconds=elapsed_before+time.time()-start,goal='ACTIVE until genuine73.5confirmation achieved; negative trial does not meet goal.')
assert result['reference_unchanged'];write(run/'resultado.json',result)
for name in ['ultima_reanudacion.pt','predicciones_avance.npz']:
 path=run/name
 if path.exists():assert path.resolve().is_relative_to(OUT.resolve());path.unlink()
print(json.dumps(dict(stage='complete',id=run.name,validation=result['validation'],best_epoch=best_epoch,epochs=len(history),saved=len(saved),seconds=result['seconds'])),flush=True)
