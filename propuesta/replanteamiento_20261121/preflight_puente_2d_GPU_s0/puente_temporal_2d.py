"""Exact Student endpoint-duration bridge in2D; all queriedurations fixed4..119.
Own conditionalmodel; not firstpassage, road-constrained motion or independentGPS-noise inference.
"""
import math
import torch

def ar_sums(rho):
 r=rho.double()
 lag=torch.arange(1,119,device=r.device,dtype=r.dtype)
 power=r[...,None].pow(lag)
 zero=torch.zeros_like(r)[...,None]
 sr=torch.cat([zero,power.cumsum(-1)],-1)
 slr=torch.cat([zero,(power*lag).cumsum(-1)],-1)
 n=torch.arange(1,120,device=r.device,dtype=r.dtype)
 return n+2*(n*sr-slr),1+sr

def mahalanobis(vector,scale,cross):
 a=vector[...,0]/scale[...,0]
 b=vector[...,1]/scale[...,1]
 return ((a-cross*b).square()+(1-cross.square())*b.square())/(1-cross.square())

def sigma_logdet(scale,cross):
 return 2*scale.log().sum(-1)+torch.log1p(-cross.square())

def endpoint_logpdf(params,df=5.):
 mu0=params['initial'].double();muc=params['cruise'].double()
 sigma=params['scale'].double();cross=params['cross'].double()
 sums,H=ar_sums(params['rho'])
 sums=sums[...,3:];H=H[...,3:]
 n=torch.arange(4,120,device=mu0.device,dtype=mu0.dtype)
 mean=n[...,None]*muc[...,None,:]+H[...,None]*(mu0-muc)[...,None,:]
 target=torch.tensor([1.,0.],device=mean.device,dtype=mean.dtype)
 quadratic=mahalanobis(target-mean,sigma[...,None,:],cross[...,None])/sums
 logdet=2*sums.log()+sigma_logdet(sigma,cross)[...,None]
 constant=math.lgamma((df+2)/2)-math.lgamma(df/2)-math.log(df*math.pi)
 return constant-.5*logdet-(df+2)/2*torch.log1p(quadratic/df)

def duration_logp(params,df=5.):
 end=endpoint_logpdf(params,df)
 prior=params['log_prior'].double().log_softmax(-1)
 energy=torch.logsumexp(prior[...,None]+end,1)
 return energy-torch.logsumexp(energy,1,keepdim=True),end

def fullpath_logpdf(increments,training_target_n,params,df=5.):
 assert increments.ndim==3 and increments.shape[-1]==2
 mu0=params['initial'].double();muc=params['cruise'].double()
 sigma=params['scale'].double();cross=params['cross'].double();rho=params['rho'].double()
 value=increments.double();n=training_target_n.double()
 first=mahalanobis(value[:,None,0]-mu0,sigma,cross)
 innovation=value[:,None,1:]-muc[...,None,:]-rho[...,None,None]*(value[:,None,:-1]-muc[...,None,:])
 q=mahalanobis(innovation,sigma[...,None,:],cross[...,None])/(1-rho.square())[...,None]
 mask=torch.arange(1,value.shape[1],device=value.device)[None]<training_target_n[:,None]
 quadratic=first+(q*mask[:,None]).sum(-1)
 logdet=n[:,None]*sigma_logdet(sigma,cross)+2*(n[:,None]-1)*torch.log1p(-rho.square())
 constant=torch.lgamma((df+2*n)/2)-math.lgamma(df/2)-n*math.log(df*math.pi)
 return constant[:,None]-.5*logdet-(df+2*n[:,None])/2*torch.log1p(quadratic/df)

def conditional_fit_logpdf(increments,training_target_n,params,end=None,df=5.):
 if end is None:end=endpoint_logpdf(params,df)
 prior=params['log_prior'].double().log_softmax(-1)
 full=fullpath_logpdf(increments,training_target_n,params,df)
 selected=end.gather(2,(training_target_n-4)[:,None,None].expand(-1,end.shape[1],1)).squeeze(2)
 return torch.logsumexp(prior+full,1)-torch.logsumexp(prior+selected,1)

def score(logp,target_n):
 cdf=logp.exp().cumsum(-1)[:,:-1]
 support=torch.arange(4,119,device=logp.device)
 target=(support[None]>=target_n[:,None]).to(cdf.dtype)
 crps=15*(cdf-target).square().sum(-1)
 nll=-logp.gather(1,(target_n-4)[:,None]).squeeze(1)
 return crps/600+.02*nll,crps,nll

def quantiles(logp):
 cdf=logp.exp().cumsum(-1)
 levels=torch.tensor([.1,.5,.9],device=logp.device,dtype=logp.dtype)
 index=(cdf[...,None]>=levels).to(torch.int64).argmax(1)
 return (index+4)*15
