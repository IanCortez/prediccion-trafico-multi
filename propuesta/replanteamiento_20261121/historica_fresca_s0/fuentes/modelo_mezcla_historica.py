"""Conditional time/geometry mixture over historical routes from other Train folds."""
import torch
from torch import nn
import torch.nn.functional as F
class HistoricalRouteMixture(nn.Module):
 def __init__(self,parent,history_mean,history_std,time_anchor,mode='historical'):
  super().__init__();self.parent=parent;self.mode=mode
  self.register_buffer('history_mean',torch.as_tensor(history_mean,dtype=torch.float32));self.register_buffer('history_std',torch.as_tensor(history_std,dtype=torch.float32))
  self.register_buffer('time_anchor',torch.as_tensor(time_anchor,dtype=torch.float32))
  self.history_head=nn.Sequential(nn.Linear(538,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,31))
  self.gate=nn.Sequential(nn.Linear(564,64),nn.SiLU(),nn.Linear(64,1))
  nn.init.zeros_(self.history_head[-1].weight);nn.init.zeros_(self.history_head[-1].bias)
  nn.init.zeros_(self.gate[-1].weight);nn.init.constant_(self.gate[-1].bias,-4)
 @property
 def duration(self):return self.parent.duration
 def branch_parameters(self):return [p for n,p in self.named_parameters() if not n.startswith('parent.')]
 def base_only(self,x,public):return self.parent(x,public)
 def historical_output(self,h,history):
  B,K,W=h.shape;H=history.float();available=H[:,:,23]>.5
  if self.mode=='parent_control':available=torch.zeros_like(available)
  z=torch.where(available[:,:,None],(H-self.history_mean)/self.history_std,0.)
  count=available.sum(1).clamp_min(1).float();mean=z.sum(1)/count[:,None]
  variance=((z-mean[:,None]).square()*available[:,:,None]).sum(1)/count[:,None]
  context=torch.cat([h,torch.cat([mean,variance.clamp_min(0).sqrt()],-1)[:,None].expand(-1,K,-1)],-1)
  gate=self.gate(context)[...,0];exists=available.any(1)[:,None]
  log_gate=torch.where(exists,F.logsigmoid(gate),torch.full_like(gate,-torch.inf));log_parent=torch.where(exists,F.logsigmoid(-gate),torch.zeros_like(gate))
  raw=self.history_head(torch.cat([h[:,:,None].expand(-1,-1,6,-1),z[:,None].expand(-1,K,-1,-1)],-1)).float()
  loc=self.time_anchor[0]+self.time_anchor[1]*(H[:,:,16]*600).clamp_min(.1).log();loc=loc[:,None]+1.5*torch.tanh(raw[...,0]/1.5)
  sigma=(.15+F.softplus(raw[...,1]-1.5)).clamp_max(1.5)
  cdf=torch.special.ndtr((self.duration.log_edges-loc[...,None])/sigma[...,None])
  probability=torch.cat([cdf[...,:1],cdf[...,1:]-cdf[...,:-1],1-cdf[...,-1:]],-1).clamp_min(1e-7);probability=probability/probability.sum(-1,keepdim=True)
  weight=(raw[...,2]+H[:,:,18].clamp_min(1e-12).log()[:,None]).masked_fill(~available[:,None],-1e9).log_softmax(-1)+log_gate[...,None]
  geo_loc=H[:,None,:,:14]+3*torch.tanh(raw[...,3:17]/3);geo_sigma=(.03+F.softplus(raw[...,17:31]-2)).clamp_max(4.)
  return weight,geo_loc,geo_sigma,probability.log(),log_parent,available.any(1)
 def forward(self,x,public,history):
  assert history.shape==(len(x),6,26)
  hidden=[];hook=self.duration.backbone.output.register_forward_pre_hook(lambda module,args:hidden.append(args[0]))
  try:base=self.parent(x,public)
  finally:hook.remove()
  assert len(hidden)==1;h=hidden[0]
  if torch.is_grad_enabled():extra=self.historical_output(h,history)
  else:
   parts=[self.historical_output(h[i:i+256],history[i:i+256]) for i in range(0,len(x),256)]
   extra=tuple(torch.cat([p[j] for p in parts],0) for j in range(6))
  cw,gl,gs,clp,log_parent,available=extra
  weights=torch.cat([base[1]+log_parent[...,None],cw],-1)
  geo_loc=torch.cat([base[2],gl],-2);geo_sigma=torch.cat([base[3],gs],-2);component=torch.cat([base[4],clp],-2)
  logp=torch.logsumexp(component+weights[...,None],dim=-2)
  logp=torch.where(available[:,None,None],logp,base[0])
  return logp,weights,geo_loc,geo_sigma,component
