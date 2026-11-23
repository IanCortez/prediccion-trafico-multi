"""Finalmatchedposition/kineticCDFgates. One80epochbudget each; no heldstopping."""
from base_replanteo import *
import torch,gc,math,shutil
from modelo_compuerta_red_cinetica import RoadCDFGate,mixture_logprob,canonical_gate,point_quantiles
from modelo_control_lognormal_puente import PublicLognormalControl,canonical_predict_control
from puente_temporal_2d import score
start=time.time();access=deny_test();attempts=[]
forbidden=(OUT/'memoria_puntos_Train_s0/puntos_Train.npz').resolve()
def no_points(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
  try:p=Path(os.fsdecode(args[0])).resolve()
  except (OSError,ValueError):return
  if p==forbidden:attempts.append(str(p));raise AssertionError('Gatefit/inference has no access to queryor sourceGPS')
sys.addaudithook(no_points)
torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(.5)
protocol=read(OUT/'protocolo_red_cinetica_Train.json');assert protocol['status']=='FROZEN_BEFORE_GRAPH_TIME_METRIC'
inputs=OUT/'indices_red_cinetica_Train_s0';proof=read(inputs/'resultado.json');assert proof['passed']
datafile=inputs/'consultas_predicciones.npz';assert sha(datafile)==proof['artifact_sha256']
run=OUT/'red_cinetica_Train_s0';assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
sources=['entrenar_compuertas_red_cinetica_Train.py','modelo_compuerta_red_cinetica.py','modelo_control_lognormal_puente.py','puente_temporal_2d.py','base_replanteo.py']
for name in sources:shutil.copy2(OUT/name,run/'fuentes'/name)
write(run/'protocolo.json',protocol)
with np.load(datafile) as z:
 X=z['publicX'];row=z['row'];timestamp=z['timestamp'];index=z['train_index'];nfit=int(z['fit_n'])
 stored_reference=z['reference_probability']
 graph_np={kind:z[kind+'_probability'] for kind in ['position','kinetic']}
 supported_np={kind:z[kind+'_supported'] for kind in graph_np}
 support_np={kind:z[kind+'_support'] for kind in graph_np}
assert nfit==40000 and X.shape==(50000,206)
y=np.load(DATA/'train_y.npy')[index].astype(np.float64);assert ((y/15).astype(np.int64)*15==y).all()
reference=read(OUT/'referencia_publica_Train_red_vial.json');file=Path(reference['model_file']);assert sha(file)==protocol['reference_model_sha256']
ck=torch.load(file,map_location='cpu',weights_only=False);baseline=PublicLognormalControl(**ck['constructor']).cuda();baseline.load_state_dict(ck['state'])
x=torch.as_tensor(X,device='cuda');lp0,q0=canonical_predict_control(baseline,x)
assert np.array_equal(lp0.exp().numpy(),stored_reference),'All50kstablelogreference exactagainst graphconstruction'
logref=lp0.to('cuda');baseline_pred=q0[:,1].numpy();del baseline,ck;gc.collect();torch.cuda.empty_cache()
target=torch.as_tensor((y/15).astype(np.int64),device='cuda')
mean=X[:nfit].mean(0,dtype=np.float64).astype(np.float32);std=np.maximum(X[:nfit].std(0,dtype=np.float64),1e-3).astype(np.float32)
constructor=dict(mean=mean,std=std)
torch.save(dict(gate_constructor=constructor,features=206,fit_n=nfit,held_n=10000,query_metadata_sha256=protocol['query_metadata_sha256'],
 input_sha256=proof['artifact_sha256'],sourcecounts_sha256=proof['sourcecounts_sha256'],reference_model_sha256=reference['model_sha256']),run/'constructor.pt')
def make():
 torch.manual_seed(protocol['gate']['seed']);return RoadCDFGate(**constructor).cuda()
def paramsha(model):
 h=hashlib.sha256()
 for name,p in model.named_parameters():h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
 return h.hexdigest()
initial=paramsha(make());histories={};metrics={};pscores={};predictions={};alllog={};allgates={};weightfiles={}
torch.cuda.reset_peak_memory_stats()
for kind in ['position','kinetic']:
 phase=time.time();sub=run/kind;sub.mkdir();model=make();assert paramsha(model)==initial
 graph=torch.as_tensor(graph_np[kind],device='cuda',dtype=torch.float64)
 mass=graph.sum(1);supported=torch.as_tensor(supported_np[kind],device='cuda')
 assert torch.all((mass[supported]-1).abs()<1e-6)
 # Normalize Float32storage rounding before the properdensitymix; no binlabel or MAE is consulted.
 graph=torch.where(supported[:,None],graph/mass.clamp_min(1e-300)[:,None],torch.zeros_like(graph))
 # Actualfinitegradients/resourcepreflight; restore exactlysameinitialmodel/optimizer afterward.
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0003);ids=torch.arange(256,device='cuda')
 for _ in range(2):
  opt.zero_grad(set_to_none=True);lp=mixture_logprob(model(x[ids]),logref[ids],graph[ids],supported[ids])
  risk,_,_=score(lp,target[ids]);loss=risk.mean();assert torch.isfinite(loss);loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
  assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);opt.step()
 assert any(p.grad is not None and bool((p.grad!=0).any()) for p in model.net[0].parameters())
 pcheck,gcheck=canonical_gate(model,x[:65],logref[:65],graph[:65],supported[:65])
 prev,grev=canonical_gate(model,x[:17].flip(0),logref[:17].flip(0),graph[:17].flip(0),supported[:17].flip(0))
 psingle,gsingle=canonical_gate(model,x[:1],logref[:1],graph[:1],supported[:1])
 assert torch.equal(pcheck[:17],prev.flip(0)) and torch.equal(pcheck[:1],psingle)
 assert torch.equal(gcheck[:17],grev.flip(0)) and torch.equal(gcheck[:1],gsingle)
 assert torch.equal(pcheck[~supported[:65].cpu()],lp0[:65][~supported[:65].cpu()])
 write(sub/'preflight_GPU.json',dict(passed=True,finite_two_step_gradients=True,firstlayer_gradient_nonzero=True,
  raw65_reverse17_singleton_logprobs_andgates_exact=True,unsupported_reference_logprob_exact=True,
  initial_parameters_sha256=initial,peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2))
 del model,opt,lp,risk,loss;pcheck=None;gc.collect()
 model=make();assert paramsha(model)==initial
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0003)
 steps=math.ceil(nfit/256);assert steps*80==12560
 schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=steps*80,eta_min=.00001);history=[]
 for epoch in range(1,81):
  model.train();order=torch.randperm(nfit,device='cuda',generator=torch.Generator(device='cuda').manual_seed(20261203+epoch))
  totals=torch.zeros(3,dtype=torch.float64,device='cuda');seen=0
  for ids in order.split(256):
   opt.zero_grad(set_to_none=True);lp=mixture_logprob(model(x[ids]),logref[ids],graph[ids],supported[ids])
   risk,crps,nll=score(lp,target[ids]);loss=risk.mean();assert torch.isfinite(loss);loss.backward()
   torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step();schedule.step()
   size=len(ids);totals+=torch.stack([loss.detach(),crps.detach().mean(),nll.detach().mean()])*size;seen+=size
  assert seen==nfit;v=(totals/seen).cpu().numpy()
  item=dict(epoch=epoch,fit_n=seen,updates=steps,loss=float(v[0]),CRPS_s=float(v[1]),bin_NLL=float(v[2]),lr=schedule.get_last_lr()[0],seconds=time.time()-phase)
  history.append(item);write(sub/'historia.json',history)
  if epoch%10==0:print(json.dumps(dict(stage='finalgate_fit',kind=kind,**item)),flush=True)
 checkpoint=sub/'epoca_80.pt';torch.save(dict(state=model.state_dict(),constructor=constructor,epoch=80,kind=kind,protocol_sha256=sha(run/'protocolo.json'),initial_parameters_sha256=initial),checkpoint)
 # Firstheldtimecomparison onlyafter fixed80epochs for eacharm.
 lp,g=canonical_gate(model,x[nfit:],logref[nfit:],graph[nfit:],supported[nfit:])
 q=point_quantiles(lp);pred=q[:,1].numpy();risk,crps,nll=score(lp,torch.as_tensor((y[nfit:]/15).astype(np.int64)))
 metrics[kind]=metric(y[nfit:],pred);pscores[kind]=dict(CRPS_s=float(crps.mean()),bin_NLL=float(nll.mean()))
 predictions[kind]=pred;alllog[kind]=lp.numpy();allgates[kind]=g.numpy();histories[kind]=history
 weightfiles[kind]=dict(file=str(checkpoint),sha256=sha(checkpoint),bytes=checkpoint.stat().st_size)
 np.savez_compressed(sub/'predicciones_held_Train.npz',row=row[nfit:],timestamp=timestamp[nfit:],true=y[nfit:],pred=pred,quantiles=q.numpy(),log_prob=lp.numpy(),gate=g.numpy(),supported=supported_np[kind][nfit:],graph_support=support_np[kind][nfit:])
 arm=dict(complete=True,metrics=metrics[kind],probability_scores=pscores[kind],weights=weightfiles[kind],
  predictions_sha256=sha(sub/'predicciones_held_Train.npz'),fit_n=nfit,held_n=10000,fixed_epochs=80,updates=12560,initial_parameters_sha256=initial,held_early_stopping=False,Validation=False,Test=False)
 write(sub/'resultado.json',arm);print(json.dumps(dict(stage='finalgate_complete',kind=kind,**arm)),flush=True)
 del model,opt,graph,supported;gc.collect();torch.cuda.empty_cache()
metrics['reference']=metric(y[nfit:],baseline_pred[nfit:])
comparisons={key:paired(y[nfit:],a,b,timestamp[nfit:]) for key,a,b in [
 ('position_vs_reference',predictions['position'],baseline_pred[nfit:]),
 ('kinetic_vs_reference',predictions['kinetic'],baseline_pred[nfit:]),
 ('kinetic_vs_position',predictions['kinetic'],predictions['position'])]}
for item in comparisons.values():item['interpretation']='Exploratory: reused10000heldTrain, oneinitialization; not primaryValidation or confirmation.'
position_pass=comparisons['position_vs_reference']['delta_mae_s']<=-5 and comparisons['position_vs_reference']['ci95_s'][1]<0
kinetic_pass=comparisons['kinetic_vs_reference']['delta_mae_s']<=-5 and comparisons['kinetic_vs_position']['delta_mae_s']<=-2 and comparisons['kinetic_vs_reference']['ci95_s'][1]<0 and comparisons['kinetic_vs_position']['ci95_s'][1]<0
archive=run/'predicciones_held_Train.npz'
np.savez_compressed(archive,row=row[nfit:],timestamp=timestamp[nfit:],true=y[nfit:],reference=baseline_pred[nfit:],**predictions)
result=dict(complete=True,scope='Finalfixedproposal: original10000reusedheldTrain; no primaryValidation or confirmation.',
 metrics=metrics,probability_scores=pscores,comparisons=comparisons,hypothesis_position_passed=bool(position_pass),hypothesis_kinetic_passed=bool(kinetic_pass),
 weights=weightfiles,predictions_sha256=sha(archive),protocol_sha256=sha(run/'protocolo.json'),constructor_sha256=sha(run/'constructor.pt'),
 initial_parameters_sha256=initial,fit_n=nfit,held_n=10000,sourceprefix_n=951718,original_15s_T_labels_andqueries_unchanged=True,
 source_count_sha256=proof['sourcecounts_sha256'],input_sha256=proof['artifact_sha256'],
 initial_gate_exactmatched=True,all_completed_results_preserved_regardless131_06=True,
 original_pointbank_open_attempts=attempts,peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,seconds=time.time()-start,
 limitations=['RepeatedlyreusedheldTrain and oneinitialization.','Sourcegraph is crossfit for fitqueries; frozenpublicreference was fitted on951718including40kfitqueries, potentially biasing fittedgate towardreference. Heldquerylabels enter neitherbaseline nor sourcekernel.','Kineticstatekernel is an approximateempiricalmodel, not an exactphysicaltrafficsimulator.'],
 user_scope='Finishthisproposal, then close. No furtheralternativefamily.',primary_best_unchanged_s=129.99003446901307,confirmation_target_s=73.5,
 Validation=False,Test=False,actual_data_files_accessed=sorted(access),sources={name:sha(run/'fuentes'/name) for name in sources})
write(run/'resultado.json',result);print((run/'resultado.json').read_text(encoding='utf-8'),flush=True)
