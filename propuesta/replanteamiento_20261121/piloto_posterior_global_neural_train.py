"""Fixed jointglobalroute experiment versus noaux/randomaux, sameinitialweights and batches, Train only."""
from base_replanteo import *
import torch,shutil,gc
from inferencia import feature_input
from modelo_aux_subviajes import duration_score
from modelo_posterior_global_neural import GlobalRoutePosterior
access=deny_test();start=time.time();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction(.35)
run=OUT/'piloto_posterior_global_neural_train_s0';assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
sources=['piloto_posterior_global_neural_train.py','modelo_posterior_global_neural.py','modelo_aux_subviajes.py','base_replanteo.py']
for name in sources:shutil.copy2(OUT/name,run/'fuentes'/name)
plan=dict(scope='Chronological40kfit/10kheldTrain raw5ODgate, not primaryValidation orconfirmation.',
 hypothesis='EndtoendmarginalT training may use globalroute labels better than separatelyfit p(C) andconditionalquantiles.',
 input='Same296publicOD+publicshortestsignature features ineverybranch. ActualTrainrouteclass teacheronly, neverquery input.',
 branches=['time_only','semantic','random'],architecture=dict(members=8,width=128,residual_blocks=3,classes=17,dropout=.1,time='Gaussianlogtimeperlatentclass; marginaldurationCDF onunchanged116bins'),
 seed=20261122,fit_n=40000,held_n=10000,epochs=100,batch=512,lr_initial=.001,lr_final=.0001,weight_decay=.0003,clip=5,
 loss='Time-onlyprimaryNLL+5RPS; semantic/random add.2routeCE+.2conditionalTclassNLL; sameparameters evenfornoaux.',
 matched='Identicalinitialweights/normalization/batchorder/dropoutseeds; randompermutation preservesexactclassfrequencies. Onlyauxlabels/loss differ.',
 evaluation='Fixedfinal100epochs ineachbranch, scoreheldonlyafterallbranches, noheldstopping/search.',
 semantic_gate='>=2seconds overstoredCatBoostcontrol withupperCI<0 AND>=1second overmatchedrandom withupperCI<0 ANDno worsethantimeonly.',
 time_only_gate='Independentlypreregistered>=2seconds overstoredCatBoostcontrol withupperCI<0.',
 data='Existingfit-onlyPCA12/KMeans16geometricresidualcodes plusunknown17th; no newcodecountsearch.',
 resources='OneGPUfraction.35, CPU4; concurrentindependentCPU2treeexperiment, no CDFdiskcache, <=100MiBnewartifacts.',Validation=False,Test=False)
write(run/'plan.json',plan)
codefolder=OUT/'codigos_ruta_global_train_s0';meta=read(codefolder/'resultado.json');assert meta['complete'];assert sha(codefolder/'codigos_y_firmas_Train.npz')==meta['outputs_sha256']
codes=dict(np.load(codefolder/'codigos_y_firmas_Train.npz'));source=OUT/'oraculo_tiempo_calle_train_s0'
with np.load(source/'auxiliares_Train.npz') as z:q={key:z[key] for key in ['row','train_index','timestamp','true']}
for key in ['row','train_index','timestamp']:assert np.array_equal(q[key],codes[key])
d,X,names=feature_input('train',read(BEST/'resultado.json'));keep=np.asarray([not n.startswith('mm_') for n in names]);assert keep.sum()==206
X=np.column_stack([X[q['train_index']][:,keep],codes['public_signature'].astype(np.float32)]);assert X.shape==(50000,296)
nf=40000;y=q['true'].astype(np.float32);mean=X[:nf].mean(0,dtype=np.float64).astype(np.float32);std=X[:nf].std(0,dtype=np.float64).clip(.01).astype(np.float32);lm=float(np.log(y[:nf].astype(np.float64)).mean())
np.savez_compressed(run/'normalizacion_Train.npz',mean=mean,std=std,log_mean=lm)
x=torch.tensor(X[:nf],device='cuda');vx=torch.tensor(X[nf:],device='cuda');target=torch.tensor(y[:nf],device='cuda')
truecode=codes['code'][:nf].astype(np.int64);randomcode=truecode[np.random.default_rng(20261122).permutation(nf)]
assert np.array_equal(np.bincount(truecode,minlength=17),np.bincount(randomcode,minlength=17))
label={'semantic':torch.tensor(truecode,device='cuda'),'random':torch.tensor(randomcode,device='cuda')}
initial_hash=None;forecasts={};histories={};checkpoints=[];class_predictions=None
@torch.no_grad()
def predict(model):
 model.eval();out=[];classes=[]
 for first in range(0,len(vx),2048):
  xx=vx[first:first+2048];n=len(xx)
  if n<2048:xx=torch.cat([xx,xx[-1:].expand(2048-n,-1)])
  lp,lw,cp=model(xx);out.append(model.quantiles(lp)[:n].cpu().numpy());classes.append(lw[:n].exp().mean(1).argmax(-1).cpu().numpy());del lp,lw,cp
 return np.concatenate(out),np.concatenate(classes)
for mode in plan['branches']:
 torch.manual_seed(20261122);torch.cuda.manual_seed_all(20261122);model=GlobalRoutePosterior(mean,std,lm)
 h=hashlib.sha256()
 for name,value in sorted(model.state_dict().items()):h.update(name.encode());h.update(value.numpy().tobytes())
 ih=h.hexdigest()
 if initial_hash is None:initial_hash=ih
 else:assert ih==initial_hash
 model=model.cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0003)
 model.train();out=model(x[:32]);pre=duration_score(out[0],target[:32],model.duration_grid)
 if mode!='time_only':pre=pre+model.joint_auxiliary(out,label[mode][:32],target[:32])
 assert torch.isfinite(pre);pre.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);model.zero_grad(set_to_none=True)
 model.eval()
 with torch.no_grad():
  lp=model(x[:17])[0];reverse=model(x[:17].flip(0))[0].flip(0);single=torch.cat([model(x[i:i+1])[0] for i in range(17)])
  assert torch.allclose(lp.exp().sum(-1),torch.ones(17,8,device='cuda'),atol=2e-6)
  assert torch.equal(model.quantiles(lp),model.quantiles(reverse)) and torch.equal(model.quantiles(lp),model.quantiles(single))
 write(run/('verificacion_'+mode+'.json'),dict(passed=True,finite_Train_gradients=True,initial_sha256=ih,reverse_singleton_quantiles_exact=True,no_route_code_in_public_forward=True,Test=False))
 history=[]
 for epoch in range(1,101):
  model.train();lr=.0001+.0009*.5*(1+np.cos(np.pi*(epoch-1)/99))
  for group in opt.param_groups:group['lr']=float(lr)
  torch.manual_seed(1000+epoch);torch.cuda.manual_seed_all(1000+epoch)
  order=torch.randperm(len(x),device='cuda',generator=torch.Generator(device='cuda').manual_seed(1000+epoch));total=0.;count=0
  for ix in order.split(512):
   opt.zero_grad(set_to_none=True);out=model(x[ix]);loss=duration_score(out[0],target[ix],model.duration_grid)
   if mode!='time_only':loss=loss+model.joint_auxiliary(out,label[mode][ix],target[ix])
   assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step();total+=float(loss.detach())*len(ix);count+=len(ix)
  row=dict(epoch=epoch,n=count,loss=total/count,lr=float(lr),seconds=time.time()-start);history.append(row)
  if epoch%10==0:
   write(run/('historia_'+mode+'.json'),history);print(json.dumps(dict(stage='Train_'+mode,**row)),flush=True)
 file=run/(mode+'.pt');torch.save(dict(state=model.state_dict(),plan=plan,mode=mode,initial_sha256=ih,constructor=dict(mean=mean,std=std,log_mean=lm)),file)
 checkpoints.append(dict(mode=mode,file=file.name,sha256=sha(file),bytes=file.stat().st_size));write(run/'etapas_completas.json',checkpoints)
 f,c=predict(model);forecasts[mode]=f;histories[mode]=history
 np.savez_compressed(run/('predicciones_'+mode+'.npz'),row=q['row'][nf:],timestamp=q['timestamp'][nf:],quantiles=f)
 if mode=='semantic':class_predictions=c
 print(json.dumps(dict(stage='completed_'+mode,epochs=100,held_forecast_computed_not_scored=True,seconds=time.time()-start)),flush=True)
 del model,opt,out,lp,reverse,single;gc.collect();torch.cuda.empty_cache()
ref=dict(np.load(source/'predicciones_held_Train.npz'))
for key in ['row','timestamp']:assert np.array_equal(ref[key],q[key][nf:])
assert np.array_equal(ref['true'],y[nf:])
metrics={mode:metric(y[nf:],f[:,1]) for mode,f in forecasts.items()}
comparison=dict(time_only_vs_reference=paired(y[nf:],forecasts['time_only'][:,1],ref['control'],q['timestamp'][nf:]),
 semantic_vs_reference=paired(y[nf:],forecasts['semantic'][:,1],ref['control'],q['timestamp'][nf:]),
 semantic_vs_random=paired(y[nf:],forecasts['semantic'][:,1],forecasts['random'][:,1],q['timestamp'][nf:]),
 semantic_vs_timeonly=paired(y[nf:],forecasts['semantic'][:,1],forecasts['time_only'][:,1],q['timestamp'][nf:]))
for v in comparison.values():v['interpretation']='Fixed100epochmatchedTrainonly globalroute study, heldonlyfinal. Not primaryValidation/confirmation.'
good=lambda v,minimum:v['delta_mae_s']<=-minimum and v['ci95_s'][1]<0
semantic_passed=bool(good(comparison['semantic_vs_reference'],2) and good(comparison['semantic_vs_random'],1) and metrics['semantic']['mae_s']<=metrics['time_only']['mae_s'])
plain_passed=bool(good(comparison['time_only_vs_reference'],2));selected='semantic' if semantic_passed else 'time_only' if plain_passed else None
np.savez_compressed(run/'predicciones_held_Train.npz',row=q['row'][nf:],timestamp=q['timestamp'][nf:],true=y[nf:],time_only=forecasts['time_only'][:,1],semantic=forecasts['semantic'][:,1],random=forecasts['random'][:,1],
 time_only_quantiles=forecasts['time_only'],semantic_quantiles=forecasts['semantic'],random_quantiles=forecasts['random'],reference=ref['control'])
result=dict(passed=selected is not None,selected=selected,metrics=metrics,reference=metric(y[nf:],ref['control']),comparisons=comparison,semantic_gate_passed=semantic_passed,time_only_gate_passed=plain_passed,
 initial_weights_sha256=initial_hash,all_branches_same_initial_weights_batches=True,held_only_final=True,epochs_per_branch=100,
 held_semantic_code_accuracy_diagnostic=float(np.mean(class_predictions==codes['code'][nf:])),no_actual_held_code_in_public_forward=True,
 sources={name:sha(run/'fuentes'/name) for name in sources},codebook_manifest_sha256=sha(codefolder/'resultado.json'),
 predictions_sha256=sha(run/'predicciones_held_Train.npz'),checkpoints=checkpoints,Validation_access=False,Test_access=False,seconds=time.time()-start,scope=plan['scope'])
write(run/'resultado.json',result);removed=[]
for checkpoint in checkpoints:
 if checkpoint['mode']==selected:continue
 p=run/checkpoint['file'];assert p.resolve().is_relative_to(run.resolve()) and sha(p)==checkpoint['sha256'];removed.append(checkpoint);p.unlink()
write(run/'pesos_descartados.json',dict(removed=removed,selected=selected,all_predictions_preserved=True))
print(json.dumps(dict(passed=result['passed'],selected=selected,metrics=metrics,reference=result['reference'],seconds=result['seconds'])),flush=True)
