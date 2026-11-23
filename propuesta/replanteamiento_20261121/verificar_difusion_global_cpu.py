"""Real neural gradients, missingaux time supervision and global-sampler units/invariance."""
from base_replanteo import *
import torch,types,math,inspect
from modelo_difusion_global import GlobalJointDiffusion
deny_test();torch.set_num_threads(2);torch.manual_seed(20261122)
mean=np.zeros(91,np.float32);mean[0]=math.log(600);std=np.ones(91,np.float32)
m=GlobalJointDiffusion(np.zeros(216),np.ones(216),mean,std,width=32)
x=torch.randn(8,216);y=torch.tensor([60,120,180,300,600,900,1200,1785],dtype=torch.float32)
phi=torch.rand(8,90);avail=torch.tensor([1,1,1,1,0,0,0,0],dtype=torch.bool)
confidence=torch.ones(8);levels=torch.full((8,),700,dtype=torch.long);noise=torch.randn(8,91)
opt=torch.optim.Adam(m.parameters(),lr=.001)
for step in range(3):
 opt.zero_grad(set_to_none=True);loss,info=m.private_loss(x,y,phi,avail,confidence,levels,noise)
 assert torch.isfinite(loss) and info['primary_examples']==8 and info['geometry_examples']==4
 loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
 assert float(m.condition[0].weight.grad.abs().sum())>0 and float(m.output[-1].weight.grad.abs().sum())>0
 opt.step()
m.eval()
real=m.quantiles(x[:3]);reverse=m.quantiles(x[:3].flip(0)).flip(0)
single=torch.cat([m.quantiles(row[None]) for row in x[:3]])
assert torch.allclose(real,reverse,atol=.01,rtol=1e-5) and torch.allclose(real,single,atol=.01,rtol=1e-5)
assert torch.isfinite(real).all() and real.min()>=60 and real.max()<=1785
# Zero velocity exposes the analytic primary-score term: its denominator remains ALL8.
with torch.no_grad():m.output[-1].weight.zero_();m.output[-1].bias.zero_()
_,withaux=m.private_loss(x,y,phi,avail,confidence,levels,noise)
_,withoutaux=m.private_loss(x,y,phi,torch.zeros_like(avail),confidence,levels,noise)
assert torch.equal(withaux['time'],withoutaux['time'])
target=(y.log()-m.target_mean[0])/m.target_std[0]
a=m.alpha_bar[levels].sqrt();s=(1-m.alpha_bar[levels]).sqrt()
expected=(a*noise[:,0]-s*target).square().mean()
assert torch.equal(withoutaux['time'],expected)
# Test reverse-process algebra and seconds with a known synthetic public-input function.
m.eval()
def context(self,features):return features
def oracle_velocity(self,z,levels,ctx):
 t=torch.tensor([120.,600.,1200.],device=z.device)[ctx[:,0].long()]
 clean=torch.zeros_like(z);clean[:,0]=(t.log()-self.target_mean[0])/self.target_std[0]
 a=self.alpha_bar[levels].sqrt()[:,None];s=(1-self.alpha_bar[levels]).sqrt()[:,None]
 return (a*z-clean)/s.clamp_min(1e-8)
m.public_context=types.MethodType(context,m);m.velocity=types.MethodType(oracle_velocity,m)
q=torch.zeros(3,216);q[:,0]=torch.arange(3)
f=m.quantiles(q);expected=torch.tensor([120.,600.,1200.])[:,None].expand(-1,3)
assert torch.allclose(f,expected,atol=.005,rtol=0),(f,expected)
r=m.quantiles(q.flip(0)).flip(0);solo=torch.cat([m.quantiles(row[None]) for row in q])
assert torch.equal(f,r) and torch.equal(f,solo)
assert torch.equal(m.base_noise[:64],m.base_noise[:256][:64])
assert not any(k in inspect.signature(m.sample_times).parameters for k in ['duration','T','phi','route','N','driver'])
try:m.sample_times(q,duration=torch.ones(3))
except TypeError:pass
else:raise AssertionError('Private label accepted at query')
assert not torch.cuda.is_initialized()
result=dict(passed=True,CPU_only=True,actual_neural_gradients_finite_nonzero=True,actual_neural_full32step_sampler_order_singleton_consistent=True,
 every_original_time_label_supervised_when_geometry_missing=True,primary_time_loss_unweighted_all_queries=True,
 reverse_process_recovers_known_public_test_function_seconds=True,reverse_singleton_exact=True,
 sample64_prefix_matches_sample256_noise=True,no_private_query_parameters=True,
 fixed32_global_denoising_steps_independentofGPS_N=True,not_a_real_performance_score=True,
 model_source_sha256=sha(OUT/'modelo_difusion_global.py'),source_sha256=sha(__file__),Test=False)
write(OUT/'verificacion_difusion_global_cpu.json',result);print(json.dumps(result),flush=True)
