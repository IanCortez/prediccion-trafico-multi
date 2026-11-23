"""Independent scipybinintegration and gradients; no training or currentheld access."""
from base_replanteo import *
import torch
from scipy.stats import lognorm
from modelo_control_lognormal_puente import folded_lognormal_logp,matched_initial_model
from puente_temporal_2d import score
start=time.time();torch.set_num_threads(2)
run=OUT/'verificacion_control_lognormal_CPU_s0';assert not run.exists();run.mkdir()
dtype=torch.float64
loc=torch.tensor([[np.log(100),np.log(600)]],dtype=dtype)
scale=torch.tensor([[.2,.4]],dtype=dtype);weight=torch.tensor([[-.2,.3]],dtype=dtype)
lp=folded_lognormal_logp(loc,scale,weight)
support=np.arange(4,120)*15;lower=support-7.5;upper=support+7.5
expected=np.zeros(116);w=weight.softmax(-1).numpy()[0]
for i in range(2):
 p=lognorm.cdf(upper,s=scale[0,i].item(),scale=np.exp(loc[0,i].item()))-lognorm.cdf(lower,s=scale[0,i].item(),scale=np.exp(loc[0,i].item()))
 p[0]=lognorm.cdf(upper[0],s=scale[0,i].item(),scale=np.exp(loc[0,i].item()))
 p[-1]=lognorm.sf(lower[-1],s=scale[0,i].item(),scale=np.exp(loc[0,i].item()))
 expected+=w[i]*p
err=float(np.max(np.abs(lp.exp().numpy()[0]-expected)));assert err<1e-13
theta=torch.tensor([np.log(300),-.8,.2],dtype=dtype,requires_grad=True)
def objective(t):
 mu=torch.stack([t[0],t[0]+.7])[None];sd=torch.stack([t[1].exp(),t[1].exp()*.8])[None]
 prior=torch.stack([t[2],-t[2]])[None]
 return score(folded_lognormal_logp(mu,sd,prior),torch.tensor([40]))[0][0]
objective(theta).backward();grad=theta.grad.numpy();finite=[]
for j in range(3):
 d=torch.zeros(3,dtype=dtype);d[j]=1e-5
 finite.append(float((objective(theta.detach()+d)-objective(theta.detach()-d))/(2e-5)))
graderr=float(np.max(np.abs(grad-finite)));assert graderr<1e-7
# Copy initialquerybackbones from the exact samefreshbridge constructor/seed, onCPUonly.
meta=torch.load(OUT/'puente_temporal_2d_Train_s0/constructor.pt',map_location='cpu',weights_only=False)
control,reference=matched_initial_model(meta['constructor'])
h=hashlib.sha256()
for key,p in reference.named_parameters():h.update(key.encode());h.update(p.detach().contiguous().numpy().tobytes())
initial=h.hexdigest();assert initial==read(OUT/'preflight_puente_2d_GPU_s0/resultado.json')['initial_parameters_sha256']
for a,b in zip(reference.members,control.members):
 for source,target in zip(a.query.parameters(),b.query.parameters()):assert torch.equal(source,target)
proof=dict(passed=True,scope='Syntheticdensity/controlinitialization only; no currentheldmetrics.',
 scipy_originalbin_probability_maxerror=err,finite_difference_gradient_maxerror=graderr,
 all_four_querybackbones_exactmatched=True,bridge_initial_parameters_sha256=initial,
 fixed116folded_originalbins=True,normalized=True,GPU_used=False,Validation=False,Test=False,
 seconds=time.time()-start,sources={n:sha(OUT/n) for n in ['modelo_control_lognormal_puente.py','verificar_control_lognormal_CPU.py','cuantiles_mezcla_balanceada.py']})
write(run/'resultado.json',proof)
for name in proof['sources']:(run/name).write_bytes((OUT/name).read_bytes())
print(json.dumps(proof))
