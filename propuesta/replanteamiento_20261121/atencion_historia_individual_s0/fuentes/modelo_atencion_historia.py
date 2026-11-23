"""Frozen22component parent plus8query-conditioned slots over64individual source trips."""
import math,torch
from torch import nn
import torch.nn.functional as F
class IndividualHistoryAttention(nn.Module):
 def __init__(self,parent,mean,std,time_anchor,mode='individual'):
  super().__init__();assert mode in ['individual','compressed'];self.parent=parent;self.mode=mode
  for p in parent.parameters():p.requires_grad_(False)
  self.register_buffer('token_mean',torch.as_tensor(mean,dtype=torch.float32));self.register_buffer('token_std',torch.as_tensor(std,dtype=torch.float32))
  self.register_buffer('time_anchor',torch.as_tensor(time_anchor,dtype=torch.float32))
  self.embed=nn.Sequential(nn.Linear(28,64),nn.SiLU(),nn.LayerNorm(64))
  self.attention_norm=nn.LayerNorm(64);self.attention=nn.MultiheadAttention(64,4,dropout=0.,batch_first=True)
  self.ff_norm=nn.LayerNorm(64);self.ff=nn.Sequential(nn.Linear(64,128),nn.SiLU(),nn.Linear(128,64))
  self.query=nn.Linear(512,64);self.slots=nn.Parameter(torch.empty(8,64));nn.init.normal_(self.slots,std=.1)
  self.head=nn.Sequential(nn.Linear(605,128),nn.SiLU(),nn.Linear(128,31))
  self.gate=nn.Sequential(nn.Linear(568,64),nn.SiLU(),nn.Linear(64,1))
  nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
  nn.init.zeros_(self.gate[-1].weight);nn.init.constant_(self.gate[-1].bias,-4.)
 @property
 def duration(self):return self.parent.duration
 def train(self,mode=True):
  super().train(mode);self.parent.eval();return self
 def branch_parameters(self):return [p for n,p in self.named_parameters() if not n.startswith('parent.')]
 def branch_output(self,hidden,raw):
  R=raw.float();B,K,W=hidden.shape
  # Full lexicographic canonicalization gives identical arithmetic for a source set.
  order=torch.arange(64,device=R.device)[None].expand(B,-1)
  for column in range(27,-1,-1):
   key=R[:,:,column].gather(1,order);permutation=torch.argsort(key,dim=1,stable=True)
   order=order.gather(1,permutation)
  R=R.gather(1,order[:,:,None].expand(-1,-1,28))
  valid=R[:,:,27]>.5;exists=valid.any(1)
  safe=valid.clone();safe[:,0]|=~exists
  z=torch.where(valid[:,:,None],(R-self.token_mean)/self.token_std,0.)
  token=self.embed(z);a=self.attention_norm(token)
  update=self.attention(a,a,a,key_padding_mask=~safe,need_weights=False)[0];token=token+update;token=token+self.ff(self.ff_norm(token))
  query=self.query(hidden)[:,:,None,:]+self.slots[None,None]
  score=torch.einsum('bkrc,bsc->bkrs',query,token)/8+R[:,None,None,:,26]
  score=score.masked_fill(~safe[:,None,None],-1e9);weight=score.softmax(-1)
  code=torch.einsum('bkrs,bsc->bkrc',weight,token)
  pooled=torch.einsum('bkrs,bsc->bkrc',weight,R)
  pooled_z=torch.einsum('bkrs,bsc->bkrc',weight,z)
  entropy=-(weight*weight.clamp_min(1e-12).log()).sum(-1,keepdim=True)/math.log(64)
  context=torch.cat([hidden[:,:,None,:].expand(-1,-1,8,-1),code,pooled_z,entropy],-1)
  raw_head=self.head(context).float()
  loc=self.time_anchor[0]+self.time_anchor[1]*pooled[...,16]+1.5*torch.tanh(raw_head[...,0]/1.5)
  sigma=(.15+F.softplus(raw_head[...,1]-1.5)).clamp_max(1.5)
  cdf=torch.special.ndtr((self.duration.log_edges-loc[...,None])/sigma[...,None])
  prob=torch.cat([cdf[...,:1],cdf[...,1:]-cdf[...,:-1],1-cdf[...,-1:]],-1).clamp_min(1e-7);prob=prob/prob.sum(-1,keepdim=True)
  count=valid.sum(1).clamp_min(1).float();mean=z.sum(1)/count[:,None]
  variance=((z-mean[:,None]).square()*valid[:,:,None]).sum(1)/count[:,None]
  summary=torch.cat([mean,variance.clamp_min(0).sqrt()],-1)
  gate=self.gate(torch.cat([hidden,summary[:,None].expand(-1,K,-1)],-1))[...,0]
  log_gate=torch.where(exists[:,None],F.logsigmoid(gate),torch.full_like(gate,-torch.inf))
  log_parent=torch.where(exists[:,None],F.logsigmoid(-gate),torch.zeros_like(gate))
  weights=raw_head[...,2].log_softmax(-1)+log_gate[...,None]
  geo_loc=pooled[...,:14]+3*torch.tanh(raw_head[...,3:17]/3)
  geo_sigma=(.03+F.softplus(raw_head[...,17:31]-2)).clamp_max(4.)
  return weights,geo_loc,geo_sigma,prob.log(),log_parent,exists
 def forward(self,x,public,history,individual):
  assert individual.shape==(len(x),64,28)
  if self.mode=='compressed':
   from historia_individual import compress_source_values
   individual=compress_source_values(individual,history)
  hidden=[];hook=self.duration.backbone.output.register_forward_pre_hook(lambda module,args:hidden.append(args[0]))
  try:
   with torch.no_grad():base=self.parent(x,public,history)
  finally:hook.remove()
  assert len(hidden)==1
  if torch.is_grad_enabled():extra=self.branch_output(hidden[0],individual)
  else:
   parts=[self.branch_output(hidden[0][i:i+256],individual[i:i+256]) for i in range(0,len(x),256)]
   extra=tuple(torch.cat([p[j] for p in parts],0) for j in range(6))
  cw,gl,gs,clp,log_parent,exists=extra
  weights=torch.cat([base[1]+log_parent[...,None],cw],-1)
  components=torch.cat([base[4],clp],-2)
  logp=torch.logsumexp(components+weights[...,None],-2)
  logp=torch.where(exists[:,None,None],logp,base[0])
  return logp,weights,torch.cat([base[2],gl],-2),torch.cat([base[3],gs],-2),components
