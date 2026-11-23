"""Public206feature encoders -> latent relaxing2DStudentvelocity parameters."""
import math,torch
from torch import nn
import torch.nn.functional as F
from modelo_mezcla_individual_mediana import ResidualBlock
class BridgeMember(nn.Module):
 def __init__(self):
  super().__init__()
  self.query=nn.Sequential(nn.Linear(206,256),nn.SiLU(),ResidualBlock(),ResidualBlock(),ResidualBlock(),nn.LayerNorm(256))
  self.head=nn.Linear(256,8*9)
  nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
 def forward(self,x,base):
  g=self.head(self.query(x)).reshape(-1,8,9)
  duration=(base+2*torch.tanh(g[...,0]/2)).clamp(math.log(15),math.log(7200)).exp()
  speed=15/duration
  cruise=torch.stack([speed,speed*torch.tanh(g[...,1])],-1)
  initial=torch.stack([speed*(1+2*torch.tanh(g[...,2])),2*speed*torch.tanh(g[...,3])],-1)
  scale=1e-5+speed[...,None]*(.05+F.softplus(g[...,4:6]-1.5))
  cross=.95*torch.tanh(g[...,6])
  # Initialrho .5; allow negative incrementcorrelation without a nonnegative-speed claim.
  rho=-.95+1.945*torch.sigmoid(g[...,7]+math.log(1.45/.495))
  return dict(initial=initial,cruise=cruise,scale=scale,cross=cross,rho=rho,log_prior=g[...,8].log_softmax(-1))
class TemporalBridge2D(nn.Module):
 def __init__(self,xmean,xstd,anchor_indices,anchor_scale,anchor_coef,members=4):
  super().__init__()
  for key,value,dtype in [('xmean',xmean,torch.float32),('xstd',xstd,torch.float32),('anchor_indices',anchor_indices,torch.long),
   ('anchor_scale',anchor_scale,torch.float32),('anchor_coef',anchor_coef,torch.float32)]:
   self.register_buffer(key,torch.as_tensor(value,dtype=dtype))
  self.members=nn.ModuleList([BridgeMember() for _ in range(members)])
 def forward(self,public_features):
  x=public_features
  normalized=((x-self.xmean)/self.xstd).clamp(-15,15)
  physical=(x[:,self.anchor_indices]*self.anchor_scale).clamp_min(.1).log()
  base=self.anchor_coef[:,0]+self.anchor_coef[:,1]*physical+torch.tensor([-.15,.15]*4,device=x.device)
  parts=[member(normalized,base) for member in self.members]
  result={key:torch.cat([part[key] for part in parts],1) for key in parts[0]}
  result['log_prior']=result['log_prior']-math.log(len(parts))
  return result
@torch.no_grad()
def canonical_predict(model,public_features):
 from puente_temporal_2d import duration_logp,quantiles
 model.eval();device=next(model.parameters()).device
 logs=[];points=[]
 for a in range(0,len(public_features),256):
  n=min(256,len(public_features)-a)
  batch=torch.zeros((256,206),device=device,dtype=torch.float32)
  batch[:n]=public_features[a:a+n].to(device)
  logp,_=duration_logp(model(batch))
  logs.append(logp[:n].cpu());points.append(quantiles(logp)[:n].cpu())
 return torch.cat(logs),torch.cat(points)
