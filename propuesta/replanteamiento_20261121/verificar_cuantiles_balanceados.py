"""Independent symmetry and finite-difference checks including separated narrow modes."""
from base_replanteo import *
import torch,math,shutil
import torch.nn.functional as F
from cuantiles_mezcla_balanceada import *
from cuantiles_mezcla_implicita import mixture_quantile
torch.set_num_threads(2);start=time.time();run=OUT/'verificacion_cuantiles_balanceados_s0'
assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
for name in ['verificar_cuantiles_balanceados.py','cuantiles_mezcla_balanceada.py']:
 shutil.copy2(OUT/name,run/'fuentes'/name)
rows=[]
for sigma in [.06,.1,.25,.5]:
 loc=torch.tensor([[math.log(100),math.log(900)]],dtype=torch.float64,requires_grad=True)
 sd=torch.full_like(loc,sigma,requires_grad=True);w=torch.full_like(loc,-math.log(2),requires_grad=True)
 true=300.;value=balanced_quantile(loc,sd,w,implicit=True);assert abs(float(value)-true)<1e-8
 dloc,dsigma,dw=torch.autograd.grad(value.sum(),(loc,sd,w))
 assert torch.allclose(dloc,torch.full_like(loc,150.),atol=1e-7,rtol=1e-7)
 expected=torch.tensor([[300*math.log(9)/(4*sigma),-300*math.log(9)/(4*sigma)]],dtype=torch.float64)
 assert torch.allclose(dsigma,expected,atol=1e-6,rtol=1e-7),(sigma,dsigma,expected)
 assert torch.isfinite(dw).all() and float(dw.abs().max())<=1e6
 rows.append(dict(sigma=sigma,median_s=float(value),dlocation=dloc.detach().numpy(),dscale=dsigma.detach().numpy(),
 weight_derivative_max=float(dw.abs().max()),naive_float32_s=float(mixture_quantile(loc.float(),sd.float(),w.float()))))
# Nondegenerate fullparameter finite differences, including normalized logweights and binNLL.
theta=torch.tensor([math.log(240),math.log(600),math.log(1050),-1.7,-1.2,-1.5,.2,-.3,.4],dtype=torch.float64,requires_grad=True)
def output(v):return v[:3][None,None],(.08+F.softplus(v[3:6]))[None,None],v[6:].log_softmax(0)[None,None]
def scalar(v,kind):
 mu,sd,w=output(v)
 if kind=='median':return balanced_quantile(mu,sd,w,implicit=True).sum()
 if kind=='nll':return -stable_bin_logprob(torch.tensor([[600.]],dtype=torch.float64),mu,sd,w).sum()
 return stable_mixture_loss((mu,sd,w),torch.tensor([700.],dtype=torch.float64))[0]
checks={}
for kind in ['median','nll','loss']:
 analytic=torch.autograd.grad(scalar(theta,kind),theta)[0].numpy();fd=[]
 for i in range(len(theta)):
  plus=theta.detach().clone();minus=theta.detach().clone();plus[i]+=1e-5;minus[i]-=1e-5
  fd.append(float((scalar(plus,kind)-scalar(minus,kind))/2e-5))
 fd=np.asarray(fd);assert np.allclose(analytic,fd,rtol=1e-5,atol=.0003),(kind,analytic,fd)
 checks[kind]=dict(max_abs_error=float(np.max(np.abs(analytic-fd))))
mu,sd,w=output(theta.detach());grid=torch.arange(60,1800,15,dtype=torch.float64)
prob=stable_bin_logprob(grid[:,None],mu.expand(116,-1,-1),sd.expand(116,-1,-1),w.expand(116,-1,-1)).exp()
assert abs(float(prob.sum())-1)<1e-12
result=dict(passed=True,deepgap_symmetric_quantiles_exact=True,deepgap_location_and_scale_gradients_exact=True,
 weight_gradient_bound_seconds=1e6,weight_gradient_limit='Numericaltrustregion only for ill-conditioned normalized-logweight derivative; location/scale derivatives retain exact log-density ratios.',
 nondegenerate_finite_differences=checks,folded_bin_mass_sum=float(prob.sum()),rows=rows,
 sources={p.name:sha(p) for p in (run/'fuentes').glob('*.py')},new_predictive_result=False,Test=False,seconds=time.time()-start)
write(run/'resultado.json',result);print(json.dumps(read(run/'resultado.json')),flush=True)
