"""Train-only detached soft targets on the existing116 duration bins."""
import torch
def teacher_probability(mu_log_seconds,log_sigma):
    mu=mu_log_seconds.detach();sigma=log_sigma.detach().exp().clamp(.05,1.5)
    edges=torch.arange(67.5,1785,15,device=mu.device,dtype=mu.dtype).log()
    cdf=torch.special.ndtr((edges[None,:]-mu[:,None])/sigma[:,None])
    p=torch.cat([cdf[:,:1],cdf[:,1:]-cdf[:,:-1],1-cdf[:,-1:]],-1).clamp_min(1e-9)
    return p/p.sum(-1,keepdim=True)
def expected_teacher_score(student_logp,mu_log_seconds,log_sigma):
    q=teacher_probability(mu_log_seconds,log_sigma)
    cross_entropy=-(q[:,None,:]*student_logp).sum(-1).mean()
    soft_rps=(student_logp.exp().cumsum(-1)-q.cumsum(-1)[:,None,:]).square().mean()
    # Omits target-only 5*mean(Fq*(1-Fq)); gradient equals expected hard NLL+5RPS.
    return cross_entropy+5*soft_rps
