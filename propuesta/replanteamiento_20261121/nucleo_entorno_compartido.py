"""Arrival under one shared finite-mixture environment for the complete trip."""
import math
import torch
from nucleo_fases import phase_arrival

def shared_environment_arrival(policy,weights,hazards,lags,end_wait,end_probability,neighbors,origin,end_states,location,sigma,nodes,prior):
    B,S,K=weights.shape
    assert hazards.shape==(B,S,K)
    assert nodes.ndim==prior.ndim==1 and len(nodes)==len(prior)
    assert torch.all(prior>0) and torch.allclose(prior.sum(),prior.new_tensor(1.))
    location=torch.as_tensor(location,device=hazards.device,dtype=hazards.dtype).expand(B)
    sigma=torch.as_tensor(sigma,device=hazards.device,dtype=hazards.dtype).expand(B)
    eta=location[:,None]+math.sqrt(2)*sigma[:,None]*nodes[None]
    Z=len(nodes)
    def repeat(value):
        return value[:,None].expand(B,Z,*value.shape[1:]).reshape(B*Z,*value.shape[1:])
    conditional_hazards=(torch.logit(hazards.clamp(1e-7,1-1e-7))[:,None]-eta[:,:,None,None]).sigmoid().reshape(B*Z,S,K)
    nb=neighbors[None].expand(B,-1,-1) if neighbors.ndim==2 else neighbors
    raw=phase_arrival(repeat(policy),repeat(weights),conditional_hazards,lags,repeat(end_wait),repeat(end_probability),repeat(nb),repeat(origin),repeat(end_states)).reshape(B,Z,-1)
    # Mix UNNORMALIZED absorption. Normalizing each node first changes the model.
    return (raw*prior[None,:,None]).sum(1)

def conditional_cohort(raw,first=4,last=120):
    cut=raw[:,first:last]
    mass=cut.sum(-1)
    return cut/mass[:,None].clamp_min(1e-12),mass
