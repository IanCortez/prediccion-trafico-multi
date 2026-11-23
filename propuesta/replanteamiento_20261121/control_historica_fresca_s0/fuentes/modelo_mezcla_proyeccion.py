"""Joint time/geometry mixture with uncertain public endpoint projection hypotheses."""
import torch
from torch import nn
import torch.nn.functional as F
class ProjectionMixture(nn.Module):
 def __init__(self,duration,route_mean,route_std,mode='expanded'):
  super().__init__();self.duration=duration;self.mode=mode
  self.register_buffer('route_mean',torch.as_tensor(route_mean,dtype=torch.float32))
  self.register_buffer('route_std',torch.as_tensor(route_std,dtype=torch.float32))
  self.register_buffer('kind',F.one_hot(torch.arange(8)%2,2).float())
  self.register_buffer('physical_anchor',duration.anchors[torch.arange(8)%2].detach().clone())
  self.candidate_head=nn.Sequential(nn.Linear(545,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,31))
  self.gate=nn.Sequential(nn.Linear(574,64),nn.SiLU(),nn.Linear(64,1))
  nn.init.zeros_(self.candidate_head[-1].weight);nn.init.zeros_(self.candidate_head[-1].bias)
  nn.init.zeros_(self.gate[-1].weight);nn.init.constant_(self.gate[-1].bias,-4)
 def branch_parameters(self):
  return [p for n,p in self.named_parameters() if not n.startswith('duration.')]
 def base_only(self,x):
  return self.duration(x)
 def candidate_output(self,h,candidates):
  B,K,W=h.shape;C=candidates.float();available=C[:,:,20]>.5
  if self.mode=='defaults':available=available.clone();available[:,2:]=False
  z=torch.where(available[:,:,None],(C-self.route_mean)/self.route_std,0.)
  count=available.sum(1).clamp_min(1).float()
  mean=z.sum(1)/count[:,None]
  variance=((z-mean[:,None]).square()*available[:,:,None]).sum(1)/count[:,None]
  summary=torch.cat([mean,variance.clamp_min(0).sqrt()],-1)
  context=torch.cat([h,summary[:,None].expand(-1,K,-1)],-1)
  gate_raw=self.gate(context)[...,0]
  log_gate=F.logsigmoid(gate_raw)
  log_gate=torch.where(available.any(1)[:,None],log_gate,torch.full_like(log_gate,-torch.inf))
  log_base=torch.where(available.any(1)[:,None],F.logsigmoid(-gate_raw),torch.zeros_like(gate_raw))
  inp=torch.cat([h[:,:,None,:].expand(-1,-1,8,-1),z[:,None].expand(-1,K,-1,-1),self.kind[None,None].expand(B,K,-1,-1)],-1)
  raw=self.candidate_head(inp).float()
  loc=self.physical_anchor[:,0]+self.physical_anchor[:,1]*(C[:,:,26]*600).clamp_min(.1).log()
  loc=loc[:,None]+1.5*torch.tanh(raw[...,0]/1.5)
  sigma=(.15+F.softplus(raw[...,1]-1.5)).clamp_max(1.5)
  cdf=torch.special.ndtr((self.duration.log_edges-loc[...,None])/sigma[...,None])
  prob=torch.cat([cdf[...,:1],cdf[...,1:]-cdf[...,:-1],1-cdf[...,-1:]],-1).clamp_min(1e-7);prob=prob/prob.sum(-1,keepdim=True)
  # GPS residual prior is fixed at30m, as in the original Train map matcher.
  prior=-.5*((C[:,:,22]*1000/30).square()+(C[:,:,23]*1000/30).square())
  weight=(raw[...,2]+prior[:,None]).masked_fill(~available[:,None],-1e9).log_softmax(-1)+log_gate[...,None]
  geo_loc=C[:,None,:,:14]+3*torch.tanh(raw[...,3:17]/3)
  geo_sigma=(.03+F.softplus(raw[...,17:31]-2)).clamp_max(4.)
  return weight,geo_loc,geo_sigma,prob.log(),log_base
 def forward(self,x,candidates):
  assert candidates.shape==(len(x),8,31)
  hidden=[];hook=self.duration.backbone.output.register_forward_pre_hook(lambda module,args:hidden.append(args[0]))
  try:base=self.duration(x)
  finally:hook.remove()
  assert len(hidden)==1;h=hidden[0]
  if torch.is_grad_enabled():
   extra=self.candidate_output(h,candidates)
  else:
   parts=[self.candidate_output(h[i:i+256],candidates[i:i+256]) for i in range(0,len(x),256)]
   extra=tuple(torch.cat([p[j] for p in parts],0) for j in range(5))
  cw,gl,gs,clp,log_base=extra
  weights=torch.cat([base[1]+log_base[...,None],cw],-1)
  geo_loc=torch.cat([base[2],gl],-2);geo_sigma=torch.cat([base[3],gs],-2)
  component_logp=torch.cat([base[4],clp],-2)
  logp=torch.logsumexp(component_logp+weights[...,None],dim=-2)
  return logp,weights,geo_loc,geo_sigma,component_logp
