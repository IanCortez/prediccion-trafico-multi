"""Each historical source retains its own conditional density before mixture aggregation."""
import math,torch
from torch import nn
import torch.nn.functional as F
class ResidualBlock(nn.Module):
 def __init__(self):
  super().__init__();self.layer=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,512),nn.SiLU(),nn.Dropout(.05),nn.Linear(512,256))
 def forward(self,x):return x+self.layer(x)/math.sqrt(3)
class Member(nn.Module):
 def __init__(self,features=206,source_features=38):
  super().__init__()
  self.query=nn.Sequential(nn.Linear(features,256),nn.SiLU(),ResidualBlock(),ResidualBlock(),ResidualBlock(),nn.LayerNorm(256))
  self.global_head=nn.Linear(256,24)
  self.source=nn.Sequential(nn.Linear(source_features,64),nn.SiLU(),nn.LayerNorm(64),nn.Linear(64,64),nn.SiLU())
  self.query_source=nn.Linear(256,64)
  self.pair=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,4))
  self.gate=nn.Sequential(nn.Linear(256+source_features*2,64),nn.SiLU(),nn.Linear(64,1))
  self.prior_strength=nn.Parameter(torch.tensor(math.log(math.expm1(.95))))
  for layer in [self.global_head,self.pair[-1],self.gate[-1]]:
   nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
 def forward(self,x,z,raw,global_base,use_sources=True):
  h=self.query(x);B=len(x)
  g=self.global_head(h).reshape(B,8,3)
  global_loc=global_base+2*torch.tanh(g[...,0]/2)
  global_sigma=(.08+F.softplus(g[...,1]-1.5)).clamp_max(1.5)
  global_weight=g[...,2].log_softmax(-1)
  if not use_sources:
   return torch.cat([global_loc,torch.zeros(B,64,device=x.device)],-1),torch.cat([global_sigma,torch.ones(B,64,device=x.device)],-1),torch.cat([global_weight,torch.full((B,64),-torch.inf,device=x.device)],-1)
  valid=raw[...,27]>.5;exists=valid.any(1);safe=valid.clone();safe[:,0]|=~exists
  source=self.source(z);query=self.query_source(h)[:,None,:].expand(-1,64,-1)
  pair=self.pair(torch.cat([source,query],-1)).float()
  location=(raw[...,16]+.8*torch.tanh(pair[...,0])).exp()+600*F.softplus(pair[...,3]-4)
  source_loc=location.clamp_min(.1).log()
  source_sigma=(.06+F.softplus(pair[...,1]-1.5)).clamp_max(1.5)
  score=pair[...,2]+(.05+F.softplus(self.prior_strength))*raw[...,26]
  score=score.masked_fill(~safe,-1e9);source_weight=score.log_softmax(-1)
  count=valid.sum(1).clamp_min(1).float();mean=z.sum(1)/count[:,None]
  variance=((z-mean[:,None]).square()*valid[...,None]).sum(1)/count[:,None]
  gate=self.gate(torch.cat([h,mean,variance.clamp_min(0).sqrt()],-1))[:,0]
  lg=torch.where(exists,F.logsigmoid(gate),torch.full_like(gate,-torch.inf))
  gb=torch.where(exists,F.logsigmoid(-gate),torch.zeros_like(gate))
  return torch.cat([global_loc,source_loc],-1),torch.cat([global_sigma,source_sigma],-1),torch.cat([global_weight+gb[:,None],source_weight+lg[:,None]],-1)
class IndividualSourceMixture(nn.Module):
 def __init__(self,xmean,xstd,rmean,rstd,anchor_indices,anchor_scale,anchor_coef,source_mode='observed',members=4):
  super().__init__();assert source_mode in ['observed','geometry','none'];self.source_mode=source_mode
  for name,value,dtype in [('xmean',xmean,torch.float32),('xstd',xstd,torch.float32),('rmean',rmean,torch.float32),('rstd',rstd,torch.float32),
    ('anchor_indices',anchor_indices,torch.long),('anchor_scale',anchor_scale,torch.float32),('anchor_coef',anchor_coef,torch.float32)]:
   self.register_buffer(name,torch.as_tensor(value,dtype=dtype))
  self.members=nn.ModuleList([Member() for _ in range(members)])
 def forward(self,x,raw):
  raw=raw.float();valid=raw[...,27]>.5
  z=torch.where(valid[...,None],((raw-self.rmean)/self.rstd).clamp(-10,10),0.)
  xx=((x-self.xmean)/self.xstd).clamp(-15,15)
  physical=(x[:,self.anchor_indices]*self.anchor_scale).clamp_min(.1).log()
  base=self.anchor_coef[:,0]+self.anchor_coef[:,1]*physical
  base=base+torch.tensor([-.15,.15]*4,device=x.device)
  parts=[member(xx,z,raw,base,self.source_mode!='none') for member in self.members]
  return tuple(torch.stack([p[k] for p in parts],1) for k in range(3))

@torch.no_grad()
def canonical_quantiles(model,x,raw):
 from cuantiles_mezcla_implicita import ensemble_quantiles
 model.eval();device=next(model.parameters()).device;parts=[]
 for a in range(0,len(x),256):
  n=min(256,len(x)-a)
  xx=torch.zeros((256,206),device=device,dtype=torch.float32)
  rr=torch.zeros((256,64,38),device=device,dtype=raw.dtype)
  xx[:n]=x[a:a+n].to(device);rr[:n]=raw[a:a+n].to(device)
  parts.append(ensemble_quantiles(model(xx,rr))[:n].cpu())
 return torch.cat(parts,0)
