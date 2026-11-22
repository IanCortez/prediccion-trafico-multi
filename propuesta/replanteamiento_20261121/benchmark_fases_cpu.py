from base_replanteo import *
from nucleo_fases import phase_arrival,geometric_wait_pmf
from nucleo_direccional import directional_arrival
import torch
torch.set_num_threads(4)
def run(kind):
 torch.manual_seed(5);B,S,J,K,H=2,512,8,8,120;D=torch.float32;nb=torch.randint(S,(B,S,J));policy=torch.randn(B,S,J,dtype=D).softmax(-1);lw=torch.randn(B,S,K,dtype=D,requires_grad=True);lh=torch.randn(B,S,K,dtype=D,requires_grad=True);ew=torch.randn(B,2,H,dtype=D).softmax(-1);ep=torch.rand(B,2);origin=torch.tensor([0,1]);ends=torch.tensor([[2,3],[4,5]]);lags=(0,0,1,1,2,2,3,3);seen={};start=time.perf_counter()
 def pack(t):
  storage=t.untyped_storage();seen[storage.data_ptr()]=storage.nbytes();return t
 with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
  w=lw.softmax(-1);h=lh.sigmoid()
  if kind=='explicit':out=directional_arrival(policy,geometric_wait_pmf(w,h,lags,H),ew,ep,nb,origin,ends)
  else:out=phase_arrival(policy,w,h,lags,ew,ep,nb,origin,ends)
  out[:,4:].sum().backward()
 assert torch.isfinite(lw.grad).all() and torch.isfinite(lh.grad).all()
 return dict(seconds=time.perf_counter()-start,unique_saved_tensor_bytes=sum(seen.values())),out.detach().numpy()
results={};outputs={}
for kind in ['explicit','phase']:
 runs=[]
 for _ in range(3):
  r,out=run(kind);runs.append(r)
 results[kind]=dict(median_forward_backward_seconds=float(np.median([r['seconds'] for r in runs])),unique_saved_tensor_bytes=runs[-1]['unique_saved_tensor_bytes']);outputs[kind]=out
report=dict(scope='CPU kernel microbenchmark,2×512states×120time bins×8neighbors, same shifted-geometric family in both algorithms',results=results,max_prediction_difference=float(np.abs(outputs['explicit']-outputs['phase']).max()),saved_tensor_storage_ratio=results['explicit']['unique_saved_tensor_bytes']/results['phase']['unique_saved_tensor_bytes'],CPU_time_ratio=results['explicit']['median_forward_backward_seconds']/results['phase']['median_forward_backward_seconds'],not_a_CUDA_measurement=True,not_actual_peak_memory=True,not_a_MAE_result=True,code_sha256=sha(__file__),Test_access=False)
write(OUT/'benchmark_fases_cpu.json',report);print(json.dumps(report))
p=OUT/'siguiente_propuesta_fases.json';plan=read(p);plan['status']='CPU model+kernel tests passed; CUDA benchmark and trainer pending';plan['Train_only_global_fit']=dict(excess_NLL_nats=.0001237895073324946,mass_within119=.9999999966317915,interpretation='Global histogram fit only; not proof of conditional fit or MAE improvement');plan['shared_trip_effect_hypothesis']=dict(evidence=read(OUT/'diagnostico_dependencia_viaje.json'),next='Consider a query-level shared latent environment ONLY after testing efficient independent-phase model. A shared environment must be marginalized over at query and learned solely with Train paths; no true query route or completed Validation trips.',references=[
 dict(title='Bayesian Dynamic Gamma Models for Route-Level Travel Time Reliability',year=2026,status='arXiv preprint,6Feb2026',url='https://arxiv.org/html/2602.07170v1',supports='Common latent environment can induce cross-segment dependence; I55 sensor route reliability, not Porto OD MAE. Our adaptation would use phase-type residence and unknown spatial paths, not reproduce its Gamma/F model.'),
 dict(title='Link Representation Learning for Probabilistic Travel Time Estimation / ProbETA',year=2025,revision='arXivv2,26Jan2026',url='https://arxiv.org/abs/2407.05895v2',doi='10.1109/TITS.2025.3590075',supports='Hierarchical link representation with intra/inter-trip dependence. Its adjacent-completed-trip query information differs from primary static OD; only the dependence mechanism is relevant here.')
 ]);write(p,plan)

