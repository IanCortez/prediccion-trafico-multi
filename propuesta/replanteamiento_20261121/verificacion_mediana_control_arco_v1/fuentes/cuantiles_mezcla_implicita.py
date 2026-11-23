"""Implicit quantile gradients for a continuous lognormal mixture with folded end bins."""
import math,torch
def cdf_logtime(logt,loc,sigma,logweight):
 z=(logt[...,None]-loc)/sigma
 return (logweight.exp()*torch.special.ndtr(z)).sum(-1)
def mixture_quantile(loc,sigma,logweight,q=.5,implicit=False,iterations=32):
 assert 0<q<1
 with torch.no_grad():
  lower=(loc-12*sigma).min(-1).values;upper=(loc+12*sigma).max(-1).values
  for _ in range(iterations):
   middle=(lower+upper)/2;prob=cdf_logtime(middle,loc,sigma,logweight)
   lower=torch.where(prob<q,middle,lower);upper=torch.where(prob<q,upper,middle)
  root=(lower+upper)/2
 if implicit and torch.is_grad_enabled():
  z=(root[...,None]-loc)/sigma
  probability=(logweight.exp()*torch.special.ndtr(z)).sum(-1)
  density=(logweight.exp()*torch.exp(-.5*z.square())/(sigma*math.sqrt(2*math.pi))).sum(-1)
  root=root-(probability-probability.detach())/density.detach().clamp_min(1e-6)
 return root.exp().clamp(60,1785)
def bin_logprob(target,loc,sigma,logweight):
 lower=(target-7.5).clamp_min(.1).log();upper=(target+7.5).log()
 pl=torch.special.ndtr((lower[...,None]-loc)/sigma)
 pu=torch.special.ndtr((upper[...,None]-loc)/sigma)
 pl=torch.where((target<=60)[...,None],torch.zeros_like(pl),pl)
 pu=torch.where((target>=1785)[...,None],torch.ones_like(pu),pu)
 return torch.logsumexp((pu-pl).clamp_min(1e-9).log()+logweight,-1)
def mixture_loss(output,target):
 loc,sigma,weight=output
 B,E,C=loc.shape
 loc=loc.reshape(B,E*C);sigma=sigma.reshape(B,E*C);weight=(weight-math.log(E)).reshape(B,E*C)
 median=mixture_quantile(loc,sigma,weight,implicit=True)
 mae=(median-target).abs().mean()/600
 nll=-bin_logprob(target,loc,sigma,weight).mean()
 return mae+.02*nll,mae,nll
def ensemble_quantiles(output,qs=(.1,.5,.9)):
 loc,sigma,weight=output
 B,E,C=loc.shape
 loc=loc.reshape(B,E*C);sigma=sigma.reshape(B,E*C);weight=(weight-math.log(E)).reshape(B,E*C)
 return torch.stack([mixture_quantile(loc,sigma,weight,q=q) for q in qs],-1)
