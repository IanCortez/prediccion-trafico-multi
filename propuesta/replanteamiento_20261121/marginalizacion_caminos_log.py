"""Exact finite-horizon sparse walk-duration partition, independent of a query's true path/N.
Research numerical instrument only; not a trained Porto model or first-passage claim.
"""
import torch

def segment_logsumexp(values, index, states):
    """Batched edge -> destination logsumexp; unreachable states have zero gradients."""
    assert values.ndim==2 and index.ndim==1 and values.shape[1]==len(index)
    ix=index[None].expand(values.shape[0],-1)
    maxima=torch.full((values.shape[0],states),-torch.inf,device=values.device,dtype=values.dtype)
    maxima=maxima.scatter_reduce(1,ix,values,reduce='amax',include_self=True)
    centers=torch.where(torch.isfinite(maxima),maxima,torch.zeros_like(maxima))
    mass=torch.zeros_like(centers).scatter_add(1,ix,torch.exp(values-centers.gather(1,ix)))
    finite=centers+torch.log(mass.clamp_min(torch.finfo(values.dtype).tiny))
    return torch.where(mass>0,finite,torch.full_like(finite,-torch.inf))

def safe_logsumexp(values,dim):
    maxima=values.amax(dim,keepdim=True)
    centers=torch.where(torch.isfinite(maxima),maxima,torch.zeros_like(maxima))
    mass=torch.exp(values-centers).sum(dim,keepdim=True)
    finite=centers+torch.log(mass.clamp_min(torch.finfo(values.dtype).tiny))
    return torch.where(mass>0,finite,torch.full_like(finite,-torch.inf)).squeeze(dim)

def walk_duration_partition(log_start,log_finish,src,dst,log_weights,min_steps=4,max_steps=119):
    """log u_n = log(a^T K^n b), retaining walks that revisit destination before ending.
    K may be substochastic or a positive finite-horizon potential. No spectral claim.
    A false reachable flag must be handled by an explicitly declared fallback; never drop rows.
    """
    assert log_start.ndim==2 and log_start.shape==log_finish.shape
    batch,states=log_start.shape
    assert log_weights.shape==(batch,len(src)) and src.shape==dst.shape
    assert 0<=min_steps<=max_steps and min_steps<=119
    assert src.dtype==dst.dtype==torch.long
    assert bool(((src>=0)&(src<states)&(dst>=0)&(dst<states)).all())
    alpha=log_start;steps=[]
    for n in range(max_steps+1):
        if n>=min_steps:steps.append(safe_logsumexp(alpha+log_finish,1))
        if n<max_steps:alpha=segment_logsumexp(alpha[:,src]+log_weights,dst,states)
    log_u=torch.stack(steps,1)
    log_z=safe_logsumexp(log_u,1);reachable=torch.isfinite(log_z)
    # Expose unreachable queries instead of returning silently invented probabilities.
    centered=log_u-torch.where(reachable,log_z,torch.zeros_like(log_z))[:,None]
    return dict(log_unnormalized=log_u,log_partition=log_z,log_prob=centered,reachable=reachable)

def ordinal_crps(log_prob,target_bin):
    assert log_prob.ndim==2 and target_bin.shape==(len(log_prob),)
    assert bool(((target_bin>=0)&(target_bin<log_prob.shape[1])).all())
    cdf=log_prob.exp().cumsum(1)[:,:-1]
    thresholds=torch.arange(log_prob.shape[1]-1,device=log_prob.device)
    truth=(thresholds[None]>=target_bin[:,None]).to(cdf.dtype)
    return ((cdf-truth)**2).mean(1)
