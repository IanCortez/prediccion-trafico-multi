"""Separate OD duration task for Train subtrips; shared encoder, original query head."""
import torch
from torch import nn
class SubtripAuxiliary(nn.Module):
 def __init__(self,duration,anchors,anchor_indices,k=16,width=512):
  super().__init__();self.duration=duration
  self.aux_weight=nn.Parameter(torch.empty(k,width,24));self.aux_bias=nn.Parameter(torch.zeros(k,24))
  nn.init.normal_(self.aux_weight,std=.005)
  self.register_buffer('aux_anchors',torch.as_tensor(anchors,dtype=torch.float32).repeat(2,1))
  self.register_buffer('aux_indices',torch.as_tensor(anchor_indices,dtype=torch.long).repeat(2))
  self.register_buffer('aux_scale',torch.tensor([600.,600.,600.,1000.]*2))
 def forward(self,x):
  return self.duration(x)
 def auxiliary(self,x):
  hidden=[]
  hook=self.duration.backbone.output.register_forward_pre_hook(lambda module,args:hidden.append(args[0]))
  try:self.duration.backbone((x[:,self.duration.keep]-self.duration.mean)/self.duration.std)
  finally:hook.remove()
  assert len(hidden)==1
  raw=(torch.einsum('bkd,kdo->bko',hidden[0],self.aux_weight)+self.aux_bias).reshape(len(x),-1,8,3).float()
  physical=x[:,self.aux_indices].float()*self.aux_scale
  base=self.aux_anchors[:,0]+self.aux_anchors[:,1]*physical.clamp_min(.1).log()
  loc=base[:,None,:]+1.5*torch.tanh(raw[...,0]/1.5)
  sigma=(.15+torch.nn.functional.softplus(raw[...,1]-1.5)).clamp_max(1.5)
  weights=raw[...,2].log_softmax(-1)
  cdf=torch.special.ndtr((self.duration.log_edges-loc[...,None])/sigma[...,None])
  prob=torch.cat([cdf[...,:1],cdf[...,1:]-cdf[...,:-1],1-cdf[...,-1:]],-1).clamp_min(1e-7)
  prob=prob/prob.sum(-1,keepdim=True)
  return torch.logsumexp(prob.log()+weights[...,None],dim=-2)
 def aux_parameters(self):
  return [self.aux_weight,self.aux_bias]
def duration_score(logprob,target,duration_grid):
 label=((target-60)/15).long()
 nll=-logprob.gather(-1,label[:,None,None].expand(-1,logprob.shape[1],1)).mean()
 desired=(duration_grid[None,None,:]>=target[:,None,None]).float()
 rps=(logprob.exp().cumsum(-1)-desired).square().mean()
 return nll+5*rps
