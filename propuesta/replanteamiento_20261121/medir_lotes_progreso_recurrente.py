"""Resource-only batching benchmark BEFORE experiment; no Validation metrics or tuning."""
from base_replanteo import *
import torch,gc
from inferencia import feature_input
from datos_progreso_recurrente import CorrectedSequences,load_graph
from modelo_progreso_recurrente import ProgressNet
from modelo_aux_subviajes import duration_score
access=deny_test();start=time.time();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
torch.cuda.set_per_process_memory_fraction(.5);torch.manual_seed(20261122)
t=CorrectedSequences();d,X,names=feature_input('train',read(BEST/'resultado.json'))
mean=X[t.eligible].mean(0,dtype=np.float64).astype(np.float32);std=X[t.eligible].std(0,dtype=np.float64).clip(.01).astype(np.float32)
m=ProgressNet(mean,std,load_graph()).cuda();x=torch.tensor(X,device='cuda');del X
a=dict(np.load(OUT/'banco_valor_vial_s0/train_proyecciones.npz'))
bank={k:torch.as_tensor(a[k].astype(np.int64) if k=='chain' else a[k],device='cuda') for k in ['chain','fraction','error_m','remaining_km','reachable']};del a
ts=torch.as_tensor(d['timestamp'],device='cuda');rng=np.random.default_rng(991);records=[]
for B in [256,512]:
 gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
 b=t.batch(rng.choice(len(t.fragments),B,replace=False));ii=b['train_index']
 torch.cuda.synchronize();tick=time.time()
 for repeat in range(3):
  m.train();m.zero_grad(set_to_none=True)
  extra,_=m.teacher_forward(x[ii],b,torch.zeros(B,4,device='cuda'))
  h=m.root(x[ii]);loss=duration_score(m.root_logprob(h),torch.as_tensor(t.y,device='cuda')[ii],m.grid)+extra
  assert torch.isfinite(loss);loss.backward()
  assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
 torch.cuda.synchronize();seconds=(time.time()-tick)/3
 records.append(dict(task='teacher_gradient',batch=B,seconds=seconds,
  peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,fragments_per_second=B/seconds))
 write(OUT/'medicion_lotes_progreso_recurrente_parcial.json',records);print(json.dumps(records[-1]),flush=True)
 m.zero_grad(set_to_none=True);del b,ii,extra,loss,h
for B in [256,512,1024]:
 gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();m.eval()
 ii=torch.as_tensor(t.eligible[:B],device='cuda')
 torch.cuda.synchronize();tick=time.time()
 out=m.public_rollout(x[ii],ts[ii],**{k:v[ii] for k,v in bank.items()})
 torch.cuda.synchronize();seconds=time.time()-tick
 assert out.shape==(B,30)
 records.append(dict(task='public_rollout_max192',batch=B,seconds=seconds,
  peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,queries_per_second=B/seconds,
  projected_allTrain_minutes=seconds*len(t.eligible)/B/60))
 del out
result=dict(complete=True,source_sha256=sha(__file__),model_sha256=sha(OUT/'modelo_progreso_recurrente.py'),
 data_sha256=sha(OUT/'datos_progreso_recurrente.py'),records=records,
 numerical_precision_and_architecture_unchanged=True,no_Validation_or_Test=True,seconds=time.time()-start)
write(OUT/'medicion_lotes_progreso_recurrente.json',result);print(json.dumps(result),flush=True)
