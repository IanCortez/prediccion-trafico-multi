"""Analytic endpoint-duration bridge with correlated signedprogress increments.
Own research instrument, not a Porto trainedmodel or a firstpassage/physicalnonnegative-speed claim.
One increment is15seconds; true queryN is never an inferenceargument.
"""
import math
import torch

def accumulated_ar_scale(rho,max_steps=119):
    """Sum of all entries of stationary AR(1) correlation for n=1..max_steps."""
    rho=rho.double()
    lag=torch.arange(1,max_steps,device=rho.device,dtype=torch.float64)
    power=rho[...,None].pow(lag)
    zero=torch.zeros_like(rho)[...,None]
    sum_r=torch.cat([zero,power.cumsum(-1)],-1)
    sum_lr=torch.cat([zero,(power*lag).cumsum(-1)],-1)
    n=torch.arange(1,max_steps+1,device=rho.device,dtype=torch.float64)
    return n+2*(n*sum_r-sum_lr)

def log_endpoint_by_duration(mean_step,scale_step,rho,df=5.,min_steps=4,max_steps=119):
    """B,K,N fixedcandidate likelihood p(s_n=1|n,style,publicq), under a global-scale t process."""
    assert mean_step.ndim==2 and mean_step.shape==scale_step.shape==rho.shape
    mu=mean_step.double();scale=scale_step.double()
    n=torch.arange(min_steps,max_steps+1,device=mu.device,dtype=torch.float64)
    variance=scale[...,None].square()*accumulated_ar_scale(rho,max_steps)[...,min_steps-1:]
    location=mu[...,None]*n
    error=(1-location).square()/variance
    const=torch.lgamma(torch.tensor((df+1)/2,device=mu.device,dtype=mu.dtype))-torch.lgamma(torch.tensor(df/2,device=mu.device,dtype=mu.dtype))-.5*math.log(df*math.pi)
    return const-.5*variance.log()-(df+1)/2*torch.log1p(error/df)

def log_duration_posterior(mean_step,scale_step,rho,log_style_prior,df=5.,min_steps=4,max_steps=119):
    """Normalize over the SAME fixed116durations for every query. Uniform prior over n.
    This is endpoint conditioning, allowing repeated earlyvisits to destination.
    The supplied publicq may condition motionparameters; no generative p(destination) claim.
    """
    prior=log_style_prior.double().log_softmax(-1)
    end=log_endpoint_by_duration(mean_step,scale_step,rho,df,min_steps,max_steps)
    energy=torch.logsumexp(prior[...,None]+end,1)
    return energy-torch.logsumexp(energy,1,keepdim=True)

def log_increment_path(increments,training_target_n,mean_step,scale_step,rho,df=5.):
    """B,K multivariate-t likelihood of supervised fitpath increments, padded after targetn.
    Cov(v_i,v_j)|lambda =lambda*scale^2*rho^|i-j|; lambda~IG(df/2,df/2).
    This sharedscale is robust at pathlevel, not an independent GPS observationerror model.
    """
    assert increments.ndim==2 and training_target_n.ndim==1
    assert increments.shape[0]==len(training_target_n)==mean_step.shape[0]
    v=increments.double();mu=mean_step.double();scale=scale_step.double();r=rho.double()
    n=training_target_n.to(dtype=torch.float64)
    first=(v[:,0,None]-mu).square()/scale.square()
    innovation=v[:,None,1:]-mu[...,None]-r[...,None]*(v[:,None,:-1]-mu[...,None])
    index=torch.arange(1,v.shape[1],device=v.device)
    valid=index[None]<training_target_n[:,None]
    quadratic=first+((innovation.square()/(scale.square()*(1-r.square()))[...,None])*valid[:,None]).sum(-1)
    logdet=n[:,None]*2*scale.log()+(n[:,None]-1)*torch.log1p(-r.square())
    dof=torch.as_tensor(df,device=v.device,dtype=v.dtype)
    constant=torch.lgamma((dof+n)/2)-torch.lgamma(dof/2)-n/2*math.log(df*math.pi)
    return constant[:,None]-.5*logdet-(df+n[:,None])/2*torch.log1p(quadratic/df)

def log_conditional_fit_path(increments,training_target_n,mean_step,scale_step,rho,log_style_prior,df=5.):
    """log p(fitinteriorprogress|targetn,endpoint=1,publicq), styles integrated exactly."""
    prior=log_style_prior.double().log_softmax(-1)
    full=log_increment_path(increments,training_target_n,mean_step,scale_step,rho,df)
    endpoint=log_endpoint_by_duration(mean_step,scale_step,rho,df)
    selected=endpoint.gather(2,(training_target_n-4)[:,None,None].expand(-1,mean_step.shape[1],1)).squeeze(2)
    return torch.logsumexp(prior+full,1)-torch.logsumexp(prior+selected,1)

def duration_score(log_prob,training_target_n):
    """Proper ordinal CRPS inseconds/600 plus.02 exactoriginalbinNLL."""
    cdf=log_prob.exp().cumsum(-1)[:,:-1]
    support=torch.arange(4,119,device=log_prob.device)
    target=(support[None]>=training_target_n[:,None]).to(cdf.dtype)
    crps_s=15*(cdf-target).square().sum(-1)
    nll=-log_prob.gather(1,(training_target_n-4)[:,None]).squeeze(1)
    return crps_s/600+.02*nll
