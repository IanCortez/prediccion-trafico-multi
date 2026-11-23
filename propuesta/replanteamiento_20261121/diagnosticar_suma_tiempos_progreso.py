"""CPU-only PRIVILEGED Train diagnostic of local-time additivity, not an OD result."""
from base_replanteo import *
import torch,gc
from datos_progreso_recurrente import CorrectedSequences,load_graph
from modelo_progreso_recurrente import ProgressNet
from inferencia import feature_input
access=deny_test();start=time.time();torch.set_num_threads(2)
run=OUT/'progreso_vial_recurrente_s0';a=read(run/'resultado_etapa_A.json');assert a['complete']
assert sha(a['model_file'])==a['model_sha256']
state=torch.load(a['model_file'],map_location='cpu',weights_only=False)
model=ProgressNet(state['state']['mean'],state['state']['std'],load_graph());model.load_state_dict(state['state']);model.eval();del state
teacher=CorrectedSequences();d,X,names=feature_input('train',read(BEST/'resultado.json'))
parents=np.unique(teacher.fragments[:,2])
selected=np.sort(np.random.default_rng(6741).choice(parents,512,replace=False))
fragments=np.flatnonzero(np.isin(teacher.fragments[:,2],selected))
mapping=np.full(len(teacher.y),-1,np.int32);mapping[selected]=np.arange(len(selected))
pred=np.zeros(512,np.float64);truth=np.zeros(512,np.float64);token_n=np.zeros(512,np.int64)
details=[];local_abs=0.;local_n=0
with torch.no_grad():
 for first in range(0,len(fragments),128):
  fi=fragments[first:first+128];b=teacher.batch(fi,device='cpu');ti=b['train_index'].numpy()
  captured=[];hook=model.time_head.register_forward_hook(lambda mod,inp,out:captured.append(out.detach()))
  _,info=model.teacher_forward(torch.as_tensor(X[ti]),b,torch.zeros(len(fi),4))
  hook.remove();assert len(captured)==1
  raw=captured[0];pace=.025+torch.nn.functional.softplus(raw[...,0])/5
  delay=torch.nn.functional.softplus(raw[...,1])*5;local=pace*b['distance_m']+delay
  valid=b['valid']
  np.add.at(pred,mapping[ti],(local*valid).sum(1).numpy())
  np.add.at(truth,mapping[ti],(b['local_s']*valid).sum(1).numpy())
  np.add.at(token_n,mapping[ti],valid.sum(1).numpy())
  local_abs+=float(torch.abs(local[valid]-b['local_s'][valid]).sum());local_n+=int(valid.sum())
  details.append({k:float(v) for k,v in info.items() if k!='style_mass'})
assert token_n.min()>0 and not torch.cuda.is_initialized()
predfile=OUT/'diagnostico_suma_tiempos_progreso_Train.npz'
np.savez_compressed(predfile,train_index=selected,row=d['row'][selected],timestamp=d['timestamp'][selected],
 original_total_T=teacher.y[selected],known_time_true=truth,known_time_pred=pred,known_token_n=token_n)
ratio=truth/teacher.y[selected]
result=dict(complete=True,scope='PRIVILEGED512Trainteacherdiagnostic, not primaryODValidation/confirmation.',
 query_not_available_fields=['actualTrainroadsequence','actualpartialpositions','observedTrainelapsedclock','recognitionposteriorwithprivatetrajectorysummary'],
 fixed_random_Train_parents=512,fragments=len(fragments),known_tokens=local_n,CPU_only=True,no_CUDA_initialized=True,
 matched_known_time_total_error=metric(truth,pred),matched_known_time_bias_s=float(np.mean(pred-truth)),
 known_time_coverage_quantiles=np.quantile(ratio,[0,.1,.5,.9,1]),local_token_MAE_s=local_abs/local_n,
 teacher_route_accuracy_unweighted_fragment_mean=float(np.mean([q['route_accuracy'] for q in details])),
 interpretation='If errorsremainlargeevenwithknownprefix/clock/road, currentlocalpseudotimeHuberdoesnotensurewholetripTconsistency. If errorssmallbutpublicfree-runpoor, route/clockexposuregap dominates. Neitherdiagnostic is a publicODscore or an errorlowerbound.',
 model_sha256=a['model_sha256'],predictions_sha256=sha(predfile),source_sha256=sha(__file__),
 no_Validation_labels_or_Test_access=True,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(OUT/'diagnostico_suma_tiempos_progreso.json',result);print(json.dumps(result),flush=True)
