"""Oneprospectivelyregistered matchedlognormal control, after the mainexperiment replay."""
from base_replanteo import *
import torch,gc,math,shutil
from datos_control_familia_puente import PublicControlInputs
from modelo_control_lognormal_puente import matched_initial_model,canonical_predict_control
from puente_temporal_2d import score,quantiles
from datos_mezcla_individual_Train import build_public206
start=time.time();access=deny_test();attempts=[]
forbidden=(OUT/'memoria_puntos_Train_s0/puntos_Train.npz').resolve()
def no_points(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
  try:p=Path(os.fsdecode(args[0])).resolve()
  except (OSError,ValueError):return
  if p==forbidden:attempts.append(str(p));raise AssertionError('Matchedpublicheadcontrol never needs originalGPSpointbank')
sys.addaudithook(no_points)
main=OUT/'puente_temporal_2d_Train_s0'
r=read(main/'resultado.json');assert r['complete'] and read(main/'reproduccion_independiente.json')['passed']
protocol=read(OUT/'protocolo_control_familia_puente_2d.json')
assert protocol['status']=='FROZEN_BEFORE_FIRST_BRIDGE_HELDRESULT' and sha(main/'protocolo.json')==protocol['main_protocol_sha256']
cpu=read(OUT/'verificacion_control_lognormal_CPU_s0/resultado.json');assert cpu['passed']
for name,digest in cpu['sources'].items():assert sha(OUT/name)==digest
run=OUT/'control_familia_puente_2d_Train_s0';assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
sources=['entrenar_control_familia_puente.py','datos_control_familia_puente.py','modelo_control_lognormal_puente.py',
 'modelo_puente_temporal_2d.py','modelo_mezcla_individual_mediana.py','puente_temporal_2d.py','cuantiles_mezcla_balanceada.py','base_replanteo.py']
for name in sources:shutil.copy2(OUT/name,run/'fuentes'/name)
write(run/'protocolo.json',protocol)
torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(.5)
data=PublicControlInputs(run)
def parameter_sha(model):
 h=hashlib.sha256()
 for name,p in model.named_parameters():h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def make():
 control,reference=matched_initial_model(data.constructor)
 assert parameter_sha(reference)==cpu['bridge_initial_parameters_sha256']
 for a,b in zip(reference.members,control.members):
  for source,target in zip(a.query.parameters(),b.query.parameters()):assert torch.equal(source,target)
 del reference
 return control.cuda()
model=make();initial=parameter_sha(model)
opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0003);ids=torch.arange(256,device='cuda')
torch.cuda.reset_peak_memory_stats()
for i in range(2):
 model.train();opt.zero_grad(set_to_none=True);lp=model(data.x[ids]);risk,_,_=score(lp,data.target_n[ids]);loss=risk.mean()
 assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
 assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);opt.step()
assert any(p.grad is not None and float(p.grad.abs().sum())>0 for p in model.members[0].query.parameters())
probe=np.linspace(0,data.nfit-1,65,dtype=int)
items=[{key:int(value[i]) if key=='timestamp' else float(value[i]) for key,value in data.query.items()} for i in probe]
raw,names=build_public206(items,read(BEST/'resultado.json')['config'])
assert names==data.names and np.array_equal(raw,data.x[torch.tensor(probe,device='cuda')].cpu().numpy())
gx=torch.tensor(raw,device='cuda')
lp,qt=canonical_predict_control(model,gx);rev,revq=canonical_predict_control(model,gx[:17].flip(0));single,singleq=canonical_predict_control(model,gx[:1])
assert torch.equal(lp[:17],rev.flip(0)) and torch.equal(lp[:1],single)
assert torch.equal(qt[:17],revq.flip(0)) and torch.equal(qt[:1],singleq)
write(run/'preflight_GPU.json',dict(passed=True,all_publicX_resident=True,finite_two_step_gradients=True,query_gradient_nonzero=True,
 raw65public206_exact=True,canonical256_reverse17_singleton_exact=True,peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,
 initial_control_parameters_sha256=initial,bridge_querybackbones_initial_exactmatched=True,no_originalGPSpointbank=True))
print(json.dumps(dict(stage='matchedhead_GPU_preflight_passed',peak_MiB=torch.cuda.max_memory_allocated()/1024**2,seconds=time.time()-start)),flush=True)
del model,opt,lp,loss,risk;gc.collect();torch.cuda.empty_cache()
model=make();assert parameter_sha(model)==initial
opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0003)
steps=math.ceil(data.nfit/256);assert steps==3718
schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=steps*4,eta_min=.00001)
history=[]
for epoch in range(1,5):
 torch.manual_seed(20261130+epoch);torch.cuda.manual_seed_all(20261130+epoch)
 order=torch.randperm(data.nfit,device='cuda',generator=torch.Generator(device='cuda').manual_seed(20261130+1000+epoch))
 model.train();totals=torch.zeros(4,device='cuda',dtype=torch.float64);count=0
 for step,ids in enumerate(order.split(256),1):
  opt.zero_grad(set_to_none=True);lp=model(data.x[ids]);n=data.target_n[ids]
  risk,crps,nll=score(lp,n);loss=risk.mean();assert torch.isfinite(loss)
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step();schedule.step()
  mae=(quantiles(lp.detach())[:,1]-n*15).abs().double().mean();size=len(ids)
  totals+=torch.stack([loss.detach(),crps.detach().mean(),nll.detach().mean(),mae])*size;count+=size
  if step%512==0:print(json.dumps(dict(stage='matchedhead_updates',epoch=epoch,step=step,total_steps=steps,seen=count,seconds=time.time()-start)),flush=True)
 assert count==data.nfit;v=totals.cpu().numpy()/count
 row=dict(epoch=epoch,train_n=count,loss=float(v[0]),train_CRPS_s=float(v[1]),train_binNLL=float(v[2]),
  train_MAE_s=float(v[3]),updates=steps,lr=schedule.get_last_lr()[0],seconds=time.time()-start)
 history.append(row);write(run/'historia.json',history);print(json.dumps(row),flush=True)
file=run/'epoca_4.pt'
torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},constructor=data.constructor,features=data.names,epoch=4,protocol=protocol),file)
lp,q=canonical_predict_control(model,data.x[data.nfit:]);logp=lp.numpy();quant=q.numpy()
saved=dict(np.load(main/'predicciones_held_Train.npz'))
for key,value in [('row',data.row[data.nfit:]),('timestamp',data.timestamp[data.nfit:]),('true',data.truth[data.nfit:])]:assert np.array_equal(saved[key],value)
old=dict(np.load(OUT/'mezcla_individual_escala_Train_s0/predicciones_held_Train.npz'))
for key in ['row','timestamp','true']:assert np.array_equal(saved[key],old[key])
truth=saved['true'];times=saved['timestamp'];metrics=metric(truth,quant[:,1]);candidate=metric(truth,saved['none'])
comparisons={key:paired(truth,saved['none'],other,times) for key,other in
 [('Student_vs_matched_lognormal',quant[:,1]),('Student_vs_saved_publicNN',old['none'])]}
for value in comparisons.values():value['interpretation']='ExploratoryreusedheldTrain; prospectivelyregistered headcomparison, oneinitialization.'
passed=all(value['delta_mae_s']<=-5 and value['ci95_s'][1]<0 for value in comparisons.values())
target=torch.tensor((truth/15).astype(np.int64));_,crps,nll=score(lp,target)
probability_scores=dict(CRPS_s=float(crps.mean()),bin_NLL=float(nll.mean()),
 coverage80=float(np.mean((truth>=quant[:,0])&(truth<=quant[:,2]))),width80_s=float((quant[:,2]-quant[:,0]).mean()))
prediction=run/'predicciones_held_Train.npz'
np.savez_compressed(prediction,row=saved['row'],timestamp=times,true=truth,pred=quant[:,1],quantiles=quant,log_prob=logp.astype(np.float32),
 Student_noaux=saved['none'],saved_publicNN=old['none'])
assert not attempts
result=dict(complete=True,secondary_head_gate_passed=bool(passed),main_auxiliary_gate_passed=bool(r['passed']),
 metrics=metrics,probability_scores=probability_scores,Student_candidate_metrics=candidate,comparisons=comparisons,
 fit_n=data.nfit,held_n=10000,fixed_epoch=4,updates=steps*4,model_file=str(file),model_sha256=sha(file),
 initial_control_parameters_sha256=initial,initial_querybackbones_exactmatched=True,independent_replay_pending=True,
 predictions_sha256=sha(prediction),constructor_sha256=sha(run/'constructor.pt'),protocol_sha256=sha(run/'protocolo.json'),
 sources={name:sha(run/'fuentes'/name) for name in sources},original_pointbank_open_attempts=attempts,
 originalqueries_labels_cohort_unchanged=True,Validation_access=False,Test_access=False,
 actual_data_files_accessed=sorted(access),seconds=time.time()-start,
 next='Independentcompleteheld/raw65control replay, then jointlyclose main/secondary hypotheses. No newepochs or densitywidth/componentgrid.')
write(run/'resultado.json',result);print((run/'resultado.json').read_text(encoding='utf-8'),flush=True)
