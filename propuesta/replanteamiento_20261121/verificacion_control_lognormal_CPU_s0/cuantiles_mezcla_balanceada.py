"""Stable signed-log-tail quantiles; does not import or change an active experiment."""
import math,torch
def _normalized(logweight):
 p=logweight.double().softmax(-1)
 return p,p.log()
def _root(loc,sigma,logweight,q,iterations=48):
 mu=loc.double();sd=sigma.double();p,lp=_normalized(logweight)
 alive=p>0
 lower=torch.where(alive,mu-12*sd,torch.full_like(mu,torch.inf)).min(-1).values
 upper=torch.where(alive,mu+12*sd,torch.full_like(mu,-torch.inf)).max(-1).values
 for _ in range(iterations):
  middle=(lower+upper)/2;z=(middle[...,None]-mu)/sd;left=z>=0
  base=(p*left).sum(-1)-q
  positive=torch.logsumexp((lp+torch.special.log_ndtr(z)).masked_fill(left,-torch.inf),-1)
  negative=torch.logsumexp((lp+torch.special.log_ndtr(-z)).masked_fill(~left,-torch.inf),-1)
  magnitude=base.abs().log()
  positive=torch.where(base>0,torch.logaddexp(positive,magnitude),positive)
  negative=torch.where(base<0,torch.logaddexp(negative,magnitude),negative)
  below=positive<negative
  lower=torch.where(below,middle,lower);upper=torch.where(below,upper,middle)
 return (lower+upper)/2,p,lp
class _ImplicitBalancedQuantile(torch.autograd.Function):
 @staticmethod
 def forward(ctx,loc,sigma,logweight,q):
  root,p,lp=_root(loc,sigma,logweight,q)
  sd=sigma.double();z=(root[...,None]-loc.double())/sd
  logdensity_part=lp-.5*z.square()-sd.log()-.5*math.log(2*math.pi)
  logdensity=torch.logsumexp(logdensity_part,-1)
  responsibility=(logdensity_part-logdensity[...,None]).exp()
  time=root.exp();active=(time>60)&(time<1785)
  ctx.save_for_backward(time,active,z,p,lp,logdensity,responsibility)
  ctx.q=q;ctx.dtypes=(loc.dtype,sigma.dtype,logweight.dtype)
  return time.clamp(60,1785)
 @staticmethod
 def backward(ctx,gradient):
  time,active,z,p,lp,logdensity,responsibility=ctx.saved_tensors
  common=gradient.double()*time*active
  dloc=common[...,None]*responsibility
  dsigma=dloc*z
  delta=torch.special.ndtr(z)-ctx.q
  # Only the ill-conditioned weight derivative is bounded; location/scale derivatives
  # use log-density responsibilities and retain their equivariance even in deep gaps.
  logabs=time.log()[...,None]+lp+delta.abs().log()-logdensity[...,None]
  weight_derivative=-delta.sign()*logabs.clamp_max(math.log(1e6)).exp()
  dw=gradient.double()[...,None]*active[...,None]*weight_derivative
  return dloc.to(ctx.dtypes[0]),dsigma.to(ctx.dtypes[1]),dw.to(ctx.dtypes[2]),None
def balanced_quantile(loc,sigma,logweight,q=.5,implicit=False):
 assert 0<q<1
 if implicit and torch.is_grad_enabled():return _ImplicitBalancedQuantile.apply(loc,sigma,logweight,float(q))
 with torch.no_grad():root,_,_=_root(loc,sigma,logweight,float(q))
 return root.exp().clamp(60,1785)
def stable_bin_logprob(target,loc,sigma,logweight):
 mu=loc.double();sd=sigma.double();target=target.double()
 lower=((target-7.5).clamp_min(.1).log()[...,None]-mu)/sd
 upper=((target+7.5).log()[...,None]-mu)/sd
 right=lower>=0
 loga=torch.where(right,torch.special.log_ndtr(-lower),torch.special.log_ndtr(upper))
 logb=torch.where(right,torch.special.log_ndtr(-upper),torch.special.log_ndtr(lower))
 gap=(logb-loga).clamp_max(0.)
 interior=loga+(-torch.expm1(gap)).clamp_min(torch.finfo(torch.float64).tiny).log()
 logprob=torch.where((target<=60)[...,None],torch.special.log_ndtr(upper),interior)
 logprob=torch.where((target>=1785)[...,None],torch.special.log_ndtr(-lower),logprob)
 return torch.logsumexp(logprob+logweight.double().log_softmax(-1),-1)
def flattened(output):
 loc,sigma,weight=output;B,E,C=loc.shape
 return loc.reshape(B,E*C),sigma.reshape(B,E*C),(weight-math.log(E)).reshape(B,E*C)
def stable_mixture_loss(output,target):
 loc,sigma,weight=flattened(output);median=balanced_quantile(loc,sigma,weight,implicit=True)
 mae=(median-target.double()).abs().mean()/600;nll=-stable_bin_logprob(target,loc,sigma,weight).mean()
 return mae+.02*nll,mae,nll
def stable_ensemble_quantiles(output,qs=(.1,.5,.9)):
 loc,sigma,weight=flattened(output)
 return torch.stack([balanced_quantile(loc,sigma,weight,q) for q in qs],-1)
@torch.no_grad()
def stable_canonical_quantiles(model,x,raw):
 model.eval();device=next(model.parameters()).device;parts=[]
 for a in range(0,len(x),256):
  n=min(256,len(x)-a);xx=torch.zeros((256,206),device=device)
  rr=torch.zeros((256,64,38),device=device,dtype=raw.dtype)
  xx[:n]=x[a:a+n].to(device);rr[:n]=raw[a:a+n].to(device)
  parts.append(stable_ensemble_quantiles(model(xx,rr))[:n].cpu())
 return torch.cat(parts,0)
