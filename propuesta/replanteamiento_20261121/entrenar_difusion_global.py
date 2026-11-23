"""Full-primary global joint time/geometry diffusion; final EMA30, no Validation selection."""
from base_replanteo import *
import torch,shutil,copy,gc,math,argparse
from inferencia import feature_input
from modelo_difusion_global import GlobalJointDiffusion,geometry_transform
ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');ap.add_argument('--random-geometry-control',action='store_true');args=ap.parse_args()
access=deny_test();start=time.time();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
torch.cuda.set_per_process_memory_fraction(.5);torch.manual_seed(20261122);torch.cuda.manual_seed_all(20261122)
proof=read(OUT/'verificacion_difusion_global_cpu.json');assert proof['passed'] and proof['model_source_sha256']==sha(OUT/'modelo_difusion_global.py')
run=OUT/('control_difusion_geometria_aleatoria_s0' if args.random_geometry_control else 'difusion_global_tiempo_geometria_s0')
if args.random_geometry_control:
 assert read(OUT/'difusion_global_tiempo_geometria_s0/resultado.json')['meaningful_gate_passed']
reference=read(OUT/'nuevo_mejor_verificado.json')
config=dict(mode='random_geometry_control' if args.random_geometry_control else 'semantic_geometry',
 seed=20261122,epochs=30,batch=4096,width=384,blocks=4,lr=.0005,final_lr=.00005,weight_decay=.0003,clip=5,
 EMA_decay=.999,loss='UnweightedoriginaltimevelocityMSE+0.2maskedgeometryvelocityMSE; geometryconfidence neverweightsprimaryT.',
 target='Joint91latent: originallogT+90observedTrainroadgeometryfeatures; noPCA/codebook/observedqueryprefix.',
 missing_geometry='Normalizedgeometryplaceholder0, geometrylossmasked; everyoriginalTstillused. Thisisauxiliarymaskedscoretraining, not an exactmissing-data likelihoodclaim.',
 query='Only216legalraw5features andfixed256SobolGaussianbase; predict64particleswith32globaldenoisingsteps. No actualroute/N/elapsed/T.',
 primary_Train=1023485,Validation=127941,Test='sealed',canonical_batch=128,default_samples=64,sensitivity_samples=256,
 selection='FixedEMAepoch30 only. No per-epochValidation orcheckpointselection.',
 gate='MAE64atleast.5secondsoverprotectedbest withupperdayCI<0; numericalcandidates additionally256particlesensitivity beforemajorclaim.',
 reference_id=reference['id'],reference_validation=reference['validation'],preservation='Allcompletedmetrics/predictions, allmodelsbelow131.06orstrictglobalbest plusdependencies.')
sources=['entrenar_difusion_global.py','modelo_difusion_global.py','base_replanteo.py']
if not args.resume:
 assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
 for n in sources:shutil.copy2(OUT/n,run/'fuentes'/n)
 write(run/'protocolo.json',config)
else:
 old=read(run/'protocolo.json')
 for k in ['reference_id','reference_validation']:config[k]=old[k]
 assert old==config
 for n in sources:assert sha(OUT/n)==sha(run/'fuentes'/n)
d,X,names=feature_input('train',read(BEST/'resultado.json'));assert len(names)==216
target=np.load(DATA/'train_y.npy');eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index']
vpublic=np.load(DATA/'validation_x.npz');assert len(eligible)==1023485 and np.max(d['timestamp'][eligible]+target[eligible])<vpublic['timestamp'].min();del vpublic
folder=OUT/'datos_firma_global90_Train_s0';meta=read(folder/'resultado.json');assert meta['complete'] and sha(folder/'firma90_Train.npz')==meta['outputs_sha256']
z=dict(np.load(folder/'firma90_Train.npz'))
for k in ['row','timestamp']:assert np.array_equal(z[k],d[k])
available=z['available'];confidence=z['teacher_confidence'];phi=z['geometry'];del z
normfile=run/'normalizacion_Train.npz'
if not args.resume:
 xmean=X[eligible].mean(0,dtype=np.float64).astype(np.float32);xstd=X[eligible].std(0,dtype=np.float64).clip(.01).astype(np.float32)
 logT=np.log(target[eligible].astype(np.float64));have=eligible[available[eligible]]
 transformed=geometry_transform(torch.as_tensor(phi[have])).numpy()
 tmean=np.r_[logT.mean(),transformed.mean(0,dtype=np.float64)].astype(np.float32)
 tstd=np.r_[logT.std(),transformed.std(0,dtype=np.float64)].clip(.01).astype(np.float32)
 del transformed
 np.savez_compressed(normfile,xmean=xmean,xstd=xstd,target_mean=tmean,target_std=tstd)
else:
 n=dict(np.load(normfile));xmean=n['xmean'];xstd=n['xstd'];tmean=n['target_mean'];tstd=n['target_std']
model=GlobalJointDiffusion(xmean,xstd,tmean,tstd).cuda()
initial_hash=hashlib.sha256()
for key,value in sorted(model.state_dict().items()):initial_hash.update(key.encode());initial_hash.update(value.cpu().numpy().tobytes())
initial_hash=initial_hash.hexdigest()
if args.random_geometry_control:assert initial_hash==read(OUT/'difusion_global_tiempo_geometria_s0/resultado.json')['initial_weights_sha256']
ema=copy.deepcopy(model).eval()
for p in ema.parameters():p.requires_grad_(False)
x=torch.tensor(X,device='cuda');yy=torch.as_tensor(target,device='cuda');pp=torch.as_tensor(phi,device='cuda');aa=torch.as_tensor(available,device='cuda');cc=torch.as_tensor(confidence,device='cuda')
del X,phi;gc.collect()
indexmap=np.arange(len(target))
if args.random_geometry_control:
 rg=np.random.default_rng(45691);have=eligible[available[eligible]];groups=np.digitize(confidence[have],[.9,.99,.99999])
 for group in np.unique(groups):
  ids=have[groups==group];indexmap[ids]=rg.permutation(ids)
mapping=torch.as_tensor(indexmap,device='cuda')
ids=torch.as_tensor(eligible,device='cuda')
opt=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
def objective(ii):return model.private_loss(x[ii],yy[ii],pp[mapping[ii]],aa[ii],cc[ii])
# Real architecture/data GPU stress preflight, before any held labels or update.
model.train();pre,details=objective(ids[:4096]);assert torch.isfinite(pre);pre.backward()
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
assert float(model.condition[0].weight.grad.abs().sum())>0
model.zero_grad(set_to_none=True)
statebytes=sum(v.numel()*v.element_size() for v in model.state_dict().values());assert statebytes<50*1024**2
write(run/'verificacion_GPU.json',dict(passed=True,real4096query_joint_loss_finite=True,conditional_encoder_gradient_nonzero=True,
 original_time_supervised_for_all_queries=True,parameters=sum(p.numel() for p in model.parameters()),state_MiB=statebytes/1024**2,
 GPU_peak_MiB=torch.cuda.max_memory_allocated()/1024**2,geometry_available=int(available[eligible].sum()),Test=False))
history=[];first=1;elapsed_before=0
if args.resume:
 ck=torch.load(run/'reanudacion.pt',map_location='cpu',weights_only=False)
 model.load_state_dict(ck['raw_state']);ema.load_state_dict(ck['ema_state']);opt.load_state_dict(ck['optimizer'])
 history=ck['history'];first=ck['epoch']+1;elapsed_before=ck['elapsed'];del ck
for epoch in range(first,31):
 model.train();torch.manual_seed(6000+epoch);torch.cuda.manual_seed_all(6000+epoch)
 order=ids[torch.randperm(len(ids),device='cuda')];n=0;ng=0;sums=np.zeros(3,np.float64);tick=time.time();steps=(len(ids)+4095)//4096
 for step,ii in enumerate(order.split(4096)):
  fraction=((epoch-1)+step/steps)/30
  lr=config['final_lr']+(config['lr']-config['final_lr'])*(.5+.5*math.cos(math.pi*fraction))
  for group in opt.param_groups:group['lr']=lr*min(1.,((epoch-1)*steps+step+1)/100)
  opt.zero_grad(set_to_none=True);loss,details=objective(ii);assert torch.isfinite(loss)
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
  with torch.no_grad():
   for e,p in zip(ema.parameters(),model.parameters()):e.lerp_(p.detach(),1-config['EMA_decay'])
  count=len(ii);sums+=np.array([float(loss.detach()),float(details['time']),float(details['geometry'])])*count
  n+=count;ng+=details['geometry_examples']
 assert n==1023485 and ng==int(available[eligible].sum())
 row=dict(epoch=epoch,primary_queries=n,geometry_labels=ng,train_score_loss=(sums/n).tolist(),seconds=time.time()-tick,elapsed=elapsed_before+time.time()-start)
 history.append(row);write(run/'historia.json',history);print(json.dumps(row),flush=True)
 torch.save(dict(raw_state=model.state_dict(),ema_state=ema.state_dict(),optimizer=opt.state_dict(),epoch=epoch,history=history,elapsed=row['elapsed']),run/'reanudacion.pt')
ema.eval();path=run/'modelo_EMA_epoca30.pt'
torch.save(dict(state=ema.state_dict(),config=config,features=names,epoch=30),path);assert path.stat().st_size<50*1024**2
del model,opt,x,pp,yy,aa,cc,mapping,ids;gc.collect();torch.cuda.empty_cache()
# First Validation labels only after finalfixedEMAweights saved.
va,V,vnames=feature_input('validation',read(BEST/'resultado.json'));truth=np.load(DATA/'validation_y.npy')
assert names==vnames and len(truth)==127941
@torch.no_grad()
def predict(X,samples=64):
 batch=128 if samples==64 else 32;parts=[]
 for first in range(0,len(X),batch):
  block=torch.as_tensor(X[first:first+batch],device='cuda');n=len(block)
  if n<batch:block=torch.cat([block,block[-1:].expand(batch-n,-1)])
  parts.append(ema.quantiles(block,samples)[:n].cpu().numpy())
 return np.concatenate(parts)
q=predict(V);m=metric(truth,q[:,1])
ref=dict(np.load(OUT/config['reference_id']/'validacion.npz'))
for k in ['row','timestamp']:assert np.array_equal(ref[k],va[k])
assert np.array_equal(ref['true'],truth)
comparison=paired(truth,q[:,1],ref['pred'],va['timestamp']);gate=bool(comparison['delta_mae_s']<=-.5 and comparison['ci95_s'][1]<0)
qualifies=m['mae_s']<131.06 or m['mae_s']<config['reference_validation']['mae_s']
np.savez_compressed(run/'validacion.npz',row=va['row'],timestamp=va['timestamp'],true=truth,pred=q[:,1],quantiles=q)
result=dict(complete=True,id=run.name,validation=m,model_file=str(path) if qualifies else None,model_sha256=sha(path),
 fixed_final_EMA_epoch=30,initial_weights_sha256=initial_hash,reference_id=config['reference_id'],paired_vs_reference=comparison,
 meaningful_gate_passed=gate,qualified=bool(qualifies),sample_count=64,source_geometry_manifest_sha256=sha(folder/'resultado.json'),
 normalization_sha256=sha(normfile),predictions_sha256=sha(run/'validacion.npz'),sources={n:sha(run/'fuentes'/n) for n in sources},
 preserved_all_primary_queries_and_labels=True,Test_access=False,actual_data_files_accessed=sorted(access),seconds=elapsed_before+time.time()-start)
write(run/'resultado.json',result)
if qualifies:
 # Same checkpoint, predeclared higher-particle sensitivity; never an observedqueryN.
 q256=predict(V,256);np.savez_compressed(run/'sensibilidad_256.npz',row=va['row'],timestamp=va['timestamp'],true=truth,pred=q256[:,1],quantiles=q256)
 write(run/'sensibilidad_256.json',dict(validation=metric(truth,q256[:,1]),same_model_sha256=result['model_sha256'],
  paired_vs_64=paired(truth,q256[:,1],q[:,1],va['timestamp']),samples=256,Test=False))
 # Actualraw5 reconstruction plus order/singleton checks on the fixed64samplepredictor.
 from consulta_od import od_records,build_od_features
 ii=np.sort(np.random.default_rng(20261122).choice(len(V),65,replace=False));fields=['o_lon','o_lat','d_lon','d_lat','timestamp']
 items=[{k:int(va[k][i]) if k=='timestamp' else float(va[k][i]) for k in fields} for i in ii]
 raw=od_records(items);RX,rnames=build_od_features(raw,read(BEST/'resultado.json')['config'])
 assert rnames==names and np.max(np.abs(RX-V[ii]))<1e-5
 rawq=predict(RX);reverse=predict(RX[::-1])[::-1];single=predict(RX[:1])
 assert np.array_equal(rawq,q[ii]) and np.array_equal(rawq,reverse) and np.array_equal(rawq[:1],single)
 for forbidden in ['route','duration','N','driver','length']:
  try:od_records([dict(items[0],**{forbidden:777})])
  except AssertionError:pass
  else:raise AssertionError('Private query field accepted')
 replay=predict(V);assert np.array_equal(replay,q)
 write(run/'certificacion_raw_y_cohorte.json',dict(passed=True,full127941_quantile_replay_exact=True,
  raw65_quantiles_exact=True,reverse_singleton_exact=True,private_fields_rejected=True,Test=False))
 write(run/'consultas_raw_sin_etiquetas.json',items)
else:
 assert path.resolve().is_relative_to(run.resolve()) and sha(path)==result['model_sha256'];path.unlink()
p=run/'reanudacion.pt'
if p.exists():assert p.resolve().is_relative_to(run.resolve());p.unlink()
print(json.dumps(dict(complete=True,validation=m,meaningful_gate=gate,qualified=qualifies,goal='ACTIVE',Test='sealed')),flush=True)
