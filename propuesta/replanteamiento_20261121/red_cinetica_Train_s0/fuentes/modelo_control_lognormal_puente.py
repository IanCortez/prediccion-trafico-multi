"""Matchedpublicbackbones and properoriginalbin scoring; conventional lognormaldensity control."""
import math,torch
from torch import nn
import torch.nn.functional as F
from modelo_mezcla_individual_mediana import ResidualBlock
from cuantiles_mezcla_balanceada import stable_bin_logprob
class LognormalMember(nn.Module):
 def __init__(self):
  super().__init__()
  self.query=nn.Sequential(nn.Linear(206,256),nn.SiLU(),ResidualBlock(),ResidualBlock(),ResidualBlock(),nn.LayerNorm(256))
  self.head=nn.Linear(256,24);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
 def forward(self,x,base):
  g=self.head(self.query(x)).reshape(-1,8,3)
  return base+2*torch.tanh(g[...,0]/2),(.08+F.softplus(g[...,1]-1.5)).clamp_max(1.5),g[...,2].log_softmax(-1)
def folded_lognormal_logp(location,scale,log_prior):
 support=torch.arange(4,120,device=location.device,dtype=torch.float64)*15
 targets=support[None].expand(location.shape[0],-1)
 logp=stable_bin_logprob(targets,location[:,None,:],scale[:,None,:],log_prior[:,None,:])
 return logp-torch.logsumexp(logp,1,keepdim=True)
class PublicLognormalControl(nn.Module):
 def __init__(self,xmean,xstd,anchor_indices,anchor_scale,anchor_coef,members=4):
  super().__init__()
  for key,value,dtype in [('xmean',xmean,torch.float32),('xstd',xstd,torch.float32),('anchor_indices',anchor_indices,torch.long),
   ('anchor_scale',anchor_scale,torch.float32),('anchor_coef',anchor_coef,torch.float32)]:
   self.register_buffer(key,torch.as_tensor(value,dtype=dtype))
  self.members=nn.ModuleList([LognormalMember() for _ in range(members)])
 def forward(self,public_features):
  x=public_features;normalized=((x-self.xmean)/self.xstd).clamp(-15,15)
  physical=(x[:,self.anchor_indices]*self.anchor_scale).clamp_min(.1).log()
  base=self.anchor_coef[:,0]+self.anchor_coef[:,1]*physical+torch.tensor([-.15,.15]*4,device=x.device)
  parts=[member(normalized,base) for member in self.members]
  loc=torch.cat([p[0] for p in parts],1);scale=torch.cat([p[1] for p in parts],1)
  prior=torch.cat([p[2]-math.log(len(parts)) for p in parts],1)
  return folded_lognormal_logp(loc,scale,prior)
def matched_initial_model(constructor,seed=20261130):
 from modelo_puente_temporal_2d import TemporalBridge2D
 torch.manual_seed(seed)
 reference=TemporalBridge2D(**constructor)
 control=PublicLognormalControl(**constructor)
 for source,target in zip(reference.members,control.members):target.query.load_state_dict(source.query.state_dict())
 return control,reference
@torch.no_grad()
def canonical_predict_control(model,public_features):
 from puente_temporal_2d import quantiles
 model.eval();device=next(model.parameters()).device;logs=[];points=[]
 for a in range(0,len(public_features),256):
  n=min(256,len(public_features)-a);x=torch.zeros((256,206),device=device,dtype=torch.float32)
  x[:n]=public_features[a:a+n].to(device);lp=model(x)
  logs.append(lp[:n].cpu());points.append(quantiles(lp)[:n].cpu())
 return torch.cat(logs),torch.cat(points)
