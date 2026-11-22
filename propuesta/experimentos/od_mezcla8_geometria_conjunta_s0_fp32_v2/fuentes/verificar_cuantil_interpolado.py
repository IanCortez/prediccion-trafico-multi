from common import *
import torch
from modelo_mezcla_rutas import RouteMixtureTabM
# Exercise the actual quantile method with synthetic probabilities, without a backbone.
class Probe:
    duration_grid=torch.arange(60,1786,15,dtype=torch.float64)
p=Probe();rng=torch.Generator().manual_seed(1729);logits=torch.randn(3,2,116,dtype=torch.float64,generator=rng,requires_grad=True);logp=logits.log_softmax(-1);q=RouteMixtureTabM.quantiles(p,logp,mode='interpolated');assert torch.all(q[:,1:]>=q[:,:-1]);assert q.min()>=60 and q.max()<=1785
loss=q[:,1].sum();grad=torch.autograd.grad(loss,logits)[0];assert torch.isfinite(grad).all();indices=[(0,0,2),(0,1,60),(1,0,45),(1,1,90),(2,0,80)];differences=[]
for idx in indices:
    upper=logits.detach().clone();lower=upper.clone();eps=1e-5;upper[idx]+=eps;lower[idx]-=eps;hi=RouteMixtureTabM.quantiles(p,upper.log_softmax(-1),(.5,),mode='interpolated').sum();lo=RouteMixtureTabM.quantiles(p,lower.log_softmax(-1),(.5,),mode='interpolated').sum();fd=float((hi-lo)/(2*eps));difference=abs(fd-float(grad[idx]));assert difference<1e-5,(idx,difference);differences.append(difference)
point=torch.full((1,1,116),-100.,dtype=torch.float64);point[0,0,36]=0.;assert abs(float(RouteMixtureTabM.quantiles(p,point.log_softmax(-1),(.5,),mode='interpolated')[0,0])-600)<1e-6
write_json(BASE/'instrumentos/cuantil_interpolado_contrato.json',dict(passed=True,finite_gradients=True,ordered_and_bounded=True,point_mass_median600s=True,max_finite_difference_error=max(differences),definition='Uniform interpolation inside duration bins, endpoints clipped to60..1785s; not claimed to recover a continuous lognormal quantile exactly.',test_labels_read=False));print('Interpolated quantile contracts passed',flush=True)
