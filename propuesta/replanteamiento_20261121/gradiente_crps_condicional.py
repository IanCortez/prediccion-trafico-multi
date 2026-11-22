"""Score-function U-statistic for CRPS of a conditionally sampled trip-time law.

Inputs MUST be IID complete trajectories accepted under the fixed cohort event,
with UNCONDITIONAL full-trajectory log probabilities. A query-label-guided sampler,
Rao-Blackwell particles, shared path noise, or a self-normalized CDF sample bank
does not meet this estimator's assumptions. P>=3 is required.
"""
import torch

def conditional_crps_weights(times,target):
    assert times.ndim==2 and target.shape==(len(times),)
    P=times.shape[1]
    assert P>=3
    a=(times-target[:,None]).abs()
    distances=(times[:,:,None]-times[:,None,:]).abs()
    rows=distances.sum(-1)
    total=rows.sum(-1,keepdim=True)
    # Covariance term for E|T-y|, then the order3 correction for .5E|T-T'|.
    weights=(a-a.mean(-1,keepdim=True))/(P-1)+(total-P*rows)/(P*(P-1)*(P-2))
    score=a.mean(-1)-total[:,0]/(2*P*(P-1))
    return weights,score

def conditional_crps_surrogate(times,target,unconditional_log_probability,scale=600.):
    assert unconditional_log_probability.shape==times.shape and scale>0
    weights,score=conditional_crps_weights(times.detach(),target.detach())
    return (weights.detach()*unconditional_log_probability).sum(-1).mean()/scale,score.detach().mean()/scale
