"""Exact finite-mixture likelihood for independent versus trip-shared environments."""
import torch,math
def environment_log_likelihood(log_weights,hazard_logits,duration,valid,lags,location,sigma,nodes,quadrature_weights,kind):
    assert kind in ('independent','shared')
    # A quadrature rule here DEFINES a finite mixture, not exact continuous integration.
    eta=location+math.sqrt(2)*sigma*nodes
    shifted=hazard_logits[:,:,None,:]-eta[None,None,:,None]
    log_h=-torch.nn.functional.softplus(-shifted)
    log_survival=-torch.nn.functional.softplus(shifted)
    exponent=duration.clamp_min(1)[:,:,None,None]-lags[None,None,None,:]-1
    component=log_weights[:,:,None,:]+log_h+exponent.clamp_min(0)*log_survival
    conditional=torch.logsumexp(component.masked_fill(exponent<0,-torch.inf),-1)
    conditional=torch.where(valid[:,:,None],conditional,torch.zeros_like(conditional))
    log_prior=quadrature_weights.log()
    if kind=='shared':return torch.logsumexp(conditional.sum(1)+log_prior[None,:],-1)
    return torch.logsumexp(conditional+log_prior[None,None,:],-1).sum(1)

