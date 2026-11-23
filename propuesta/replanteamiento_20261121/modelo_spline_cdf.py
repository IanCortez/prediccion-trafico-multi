"""Monotone conditional splineCDF on unchanged Porto duration bins.
Rational-quadratic spline equations: Durkan etal, NeuralSplineFlows2019.
Reference implementation reviewed: https://github.com/bayesiains/nflows (MIT).
This batchedCDF implementation does not import or install nflows.
"""
import math,torch
from torch import nn
from torch.nn import functional as F
def spline_transform(value,raw,inverse=False,bins=32,bound=4.):
 widths=.001+(1-.001*bins)*raw[...,:bins].softmax(-1)
 heights=.001+(1-.001*bins)*raw[...,bins:2*bins].softmax(-1)
 ends=torch.ones_like(widths[...,:1])
 cw=torch.cat([-bound*ends,-bound+2*bound*widths.cumsum(-1)[...,:-1],bound*ends],-1)
 ch=torch.cat([-bound*ends,-bound+2*bound*heights.cumsum(-1)[...,:-1],bound*ends],-1)
 widths=cw[...,1:]-cw[...,:-1];heights=ch[...,1:]-ch[...,:-1]
 identity_offset=math.log(math.expm1(1-.001))
 deriv=torch.cat([ends,.001+F.softplus(raw[...,2*bins:]+identity_offset),ends],-1)
 inside=(value>=-bound)&(value<=bound);v=value.clamp(-bound,bound).contiguous()
 knots=ch if inverse else cw
 index=torch.searchsorted(knots.contiguous(),v,right=True).sub(1).clamp(0,bins-1)
 x0=cw.gather(-1,index);y0=ch.gather(-1,index);w=widths.gather(-1,index);h=heights.gather(-1,index)
 d0=deriv.gather(-1,index);d1=deriv.gather(-1,index+1);delta=h/w;curvature=d0+d1-2*delta
 if inverse:
  diff=v-y0;a=diff*curvature+h*(delta-d0);b=h*d0-diff*curvature;c=-delta*diff
  discriminant=(b.square()-4*a*c).clamp_min(torch.finfo(v.dtype).tiny)
  theta=(2*c/(-b-discriminant.sqrt())).clamp(0,1);out=x0+theta*w
 else:
  theta=((v-x0)/w).clamp(0,1);mixed=theta*(1-theta)
  out=y0+h*(delta*theta.square()+d0*mixed)/(delta+curvature*mixed)
 return torch.where(inside,out,value)
class EnsembleLinear(nn.Module):
 def __init__(self,k,inputs,outputs):
  super().__init__();self.weight=nn.Parameter(torch.empty(k,inputs,outputs));self.bias=nn.Parameter(torch.zeros(k,outputs))
  nn.init.uniform_(self.weight,-1/math.sqrt(inputs),1/math.sqrt(inputs))
 def forward(self,x):return torch.einsum('bki,kio->bko',x,self.weight)+self.bias
class SplineDurationCDF(nn.Module):
 def __init__(self,mean,std,log_mean,log_std,k=8,width=192,bins=32):
  super().__init__();self.k=k;self.bins=bins;self.width=width
  self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32));self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32))
  self.register_buffer('log_mean',torch.as_tensor(log_mean,dtype=torch.float32));self.register_buffer('log_std',torch.as_tensor(log_std,dtype=torch.float32))
  f=len(mean);self.feature_w1=nn.Parameter(torch.empty(f,8));self.feature_b1=nn.Parameter(torch.zeros(f,8))
  self.feature_w2=nn.Parameter(torch.empty(f,8,3));self.feature_b2=nn.Parameter(torch.zeros(f,3))
  nn.init.normal_(self.feature_w1,std=.3);nn.init.normal_(self.feature_w2,std=.3)
  self.input=EnsembleLinear(k,4*f,width)
  self.norms=nn.ModuleList([nn.LayerNorm(width) for i in range(3)])
  self.up=nn.ModuleList([EnsembleLinear(k,width,2*width) for i in range(3)])
  self.down=nn.ModuleList([EnsembleLinear(k,2*width,width) for i in range(3)])
  self.output=EnsembleLinear(k,width,3*bins-1);self.final_norm=nn.LayerNorm(width)
  nn.init.normal_(self.output.weight,std=.005)
  grid=torch.arange(60,1800,15,dtype=torch.float32);self.register_buffer('duration_grid',grid);self.register_buffer('log_edges',(grid[:-1]+7.5).log())
 def condition(self,x):
  z=(x-self.mean)/self.std
  e=F.silu(z[...,None]*self.feature_w1+self.feature_b1);e=torch.einsum('bfi,fij->bfj',e,self.feature_w2)+self.feature_b2
  encoded=torch.cat([z,e.flatten(1)],-1)[:,None].expand(-1,self.k,-1);h=F.silu(self.input(encoded))
  for norm,up,down in zip(self.norms,self.up,self.down):
   update=F.silu(up(norm(h)));update=F.dropout(update,p=.1,training=self.training);h=h+F.dropout(down(update),p=.1,training=self.training)
  return self.output(self.final_norm(h))
 def forward(self,x):
  raw=self.condition(x);edges=((self.log_edges-self.log_mean)/self.log_std)[None,None].expand(len(x),self.k,-1)
  z=spline_transform(edges.double(),raw.double(),inverse=True,bins=self.bins);cdf=torch.special.ndtr(z)
  p=torch.cat([cdf[...,:1],cdf[...,1:]-cdf[...,:-1],1-cdf[...,-1:]],-1).clamp_min(1e-7);p=p/p.sum(-1,keepdim=True)
  return p.log().float()
 @torch.no_grad()
 def quantiles(self,lp):
  cdf=lp.exp().mean(1).cumsum(-1);q=torch.tensor([.1,.5,.9],device=lp.device);index=(cdf[:,:,None]>=q).float().argmax(1)
  return self.duration_grid[index]
