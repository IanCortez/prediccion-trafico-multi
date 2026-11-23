"""Actualall-data residentpreflight, fitgradients and rawfivefield/order checks only."""
from base_replanteo import *
import torch,gc
from datos_puente_temporal_2d import BridgeInputs
from modelo_puente_temporal_2d import TemporalBridge2D,canonical_predict
from puente_temporal_2d import duration_logp,score,conditional_fit_logpdf
from datos_mezcla_individual_Train import build_public206
start=time.time();access=deny_test()
run=OUT/'preflight_puente_2d_GPU_s0';assert not run.exists();run.mkdir()
assert read(OUT/'verificacion_puente_2d_CPU_s0/resultado.json')['passed']
assert read(OUT/'verificacion_coordenadas_2d_Train_s0/resultado.json')['passed']
torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(.5)
data=BridgeInputs(run);torch.manual_seed(20261130);torch.cuda.manual_seed_all(20261130)
model=TemporalBridge2D(**data.constructor).cuda()
h=hashlib.sha256()
for key,p in model.named_parameters():h.update(key.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
initial=h.hexdigest()
torch.cuda.reset_peak_memory_stats();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0003)
ids=torch.arange(256,device='cuda');gradient_norms=[]
for i in range(2):
 model.train();opt.zero_grad(set_to_none=True);delta,n=data.fit_targets(ids,'observed')
 params=model(data.x[ids]);logp,end=duration_logp(params)
 risk,crps,nll=score(logp,n);aux=-conditional_fit_logpdf(delta,n,params,end)/(2*(n-1))
 loss=(risk+.02*aux).mean();assert torch.isfinite(loss);loss.backward()
 torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
 assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 gradient_norms.append(float(sum(p.grad.abs().sum() for p in model.members[0].query.parameters() if p.grad is not None)))
 opt.step()
assert gradient_norms[-1]>0
for member in model.members:
 g=member.head.weight.grad.reshape(8,9,-1)
 assert bool((g[:,4:8].abs().sum((0,2))>0).all()),'Covariance/correlation gradients required'
# Sourcecoordinate equality against independentlystored CPUpositions on the SAME fitparents.
sample=dict(np.load(OUT/'verificacion_coordenadas_2d_Train_s0/coordenadas_ajuste4096.npz'))
inverse=np.full(1023529,-1,np.int32);inverse[data.index[:data.nfit]]=np.arange(data.nfit)
probe=inverse[sample['parent'][:65]];assert (probe>=0).all()
actual,n=data.fit_targets(probe,'observed');expected=sample['private_fit_positions'][:65].copy()
for i,count in enumerate(sample['private_fit_n'][:65]):expected[i,int(count)+1:]=[1,0]
coordinate_error=float(np.max(np.abs(actual.cpu().numpy()-np.diff(expected,axis=1))));assert coordinate_error<1e-6
try:data.fit_targets(np.array([data.nfit]),'observed')
except AssertionError:held_guard=True
else:raise AssertionError('HeldGPS accessguard failed')
query={key:value[probe] for key,value in data.query.items()}
items=[{key:int(value[i]) if key=='timestamp' else float(value[i]) for key,value in query.items()} for i in range(65)]
raw,names=build_public206(items,read(BEST/'resultado.json')['config'])
assert names==data.names and np.array_equal(raw,data.x[torch.tensor(probe,device='cuda')].cpu().numpy())
gx=torch.tensor(raw,device='cuda')
p,q=canonical_predict(model,gx);reverse,reverseq=canonical_predict(model,gx[:17].flip(0));single,singleq=canonical_predict(model,gx[:1])
assert torch.equal(p[:17],reverse.flip(0)) and torch.equal(p[:1],single)
assert torch.equal(q[:17],reverseq.flip(0)) and torch.equal(q[:1],singleq)
proof=dict(passed=True,fit_n=data.nfit,resident_all_data=True,initial_parameters_sha256=initial,
 finite_two_step_gradients=True,query_gradient_norms=gradient_norms,covariance_correlation_gradients_nonzero=True,
 fitcoordinate_CPU_GPU_maxerror=coordinate_error,heldGPS_guard_passed=held_guard,
 raw65public206_exact=True,canonical256_reverse17_singleton_exact=True,
 peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,
 new_population_cache_written=False,weights_saved=False,Validation=False,Test=False,
 actual_data_files_accessed=sorted(access),seconds=time.time()-start,
 sources={name:sha(OUT/name) for name in ['preflight_puente_2d_GPU.py','modelo_puente_temporal_2d.py','datos_puente_temporal_2d.py','puente_temporal_2d.py','coordenadas_trayecto_2d.py']})
write(run/'resultado.json',proof)
for name in proof['sources']:(run/name).write_bytes((OUT/name).read_bytes())
print(json.dumps(proof))
