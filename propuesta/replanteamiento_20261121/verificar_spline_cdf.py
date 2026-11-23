"""Meaningful spline inversion/monotonicity/gradient and OD probability invariants."""
from base_replanteo import *
import torch
from modelo_spline_cdf import SplineDurationCDF,spline_transform
access=deny_test();start=time.time();torch.set_num_threads(2);torch.manual_seed(20261122)
raw=torch.randn(2,3,95,dtype=torch.float64)*.3;grid=torch.linspace(-6,6,1001,dtype=torch.float64)[None,None].expand(2,3,-1)
mapped=spline_transform(grid,raw);back=spline_transform(mapped,raw,inverse=True)
error=float((back-grid).abs().max());assert error<1e-8 and (mapped[...,1:]-mapped[...,:-1]>0).all()
identity=spline_transform(grid,torch.zeros_like(raw));identity_error=float((identity-grid).abs().max());assert identity_error<1e-10
x=torch.randn(17,206);model=SplineDurationCDF(torch.zeros(206),torch.ones(206),6.,.65);model.eval()
with torch.no_grad():
 p=model(x);rev=model(x.flip(0)).flip(0);single=torch.cat([model(x[i:i+1]) for i in range(17)])
 assert torch.isfinite(p).all() and torch.allclose(p.exp().sum(-1),torch.ones(17,8),atol=2e-6)
 assert torch.equal(model.quantiles(p),model.quantiles(rev)) and torch.equal(model.quantiles(p),model.quantiles(single))
 assert (model.quantiles(p)[:,1:]>=model.quantiles(p)[:,:-1]).all()
 maxdiff=float((p.exp()-single.exp()).abs().max());assert maxdiff<2e-6
model.train();lp=model(x);target=torch.arange(17)%116;loss=-lp.gather(-1,target[:,None,None].expand(-1,8,1)).mean();loss.backward()
assert all(torch.isfinite(q.grad).all() for q in model.parameters() if q.grad is not None)
result=dict(passed=True,inverse_error_float64=error,identity_error_float64=identity_error,strict_monotonicity_including_linear_tails=True,
 probability_normalization=True,reverse_singleton_allquantiles_exact=True,max_singleton_probability_error=maxdiff,
 finite_NLL_gradients=True,parameters=sum(q.numel() for q in model.parameters()),model_sha256=sha(OUT/'modelo_spline_cdf.py'),
 datasets_opened=False,Test_access=False,seconds=time.time()-start)
write(OUT/'verificacion_spline_cdf.json',result);print(json.dumps(result),flush=True)
