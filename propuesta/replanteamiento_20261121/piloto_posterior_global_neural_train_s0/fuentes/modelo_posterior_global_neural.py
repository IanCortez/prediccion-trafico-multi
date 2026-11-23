"""Joint marginalduration and semanticwhole-route latentcategorical model. Publicforward accepts x only."""
import math,torch
from torch import nn
from torch.nn import functional as F
class BatchedLinear(nn.Module):
 def __init__(self,k,i,o):
  super().__init__();self.w=nn.Parameter(torch.empty(k,i,o));self.b=nn.Parameter(torch.zeros(k,o));nn.init.uniform_(self.w,-1/math.sqrt(i),1/math.sqrt(i))
 def forward(self,x):return torch.einsum('bki,kio->bko',x,self.w)+self.b
class GlobalRoutePosterior(nn.Module):
 def __init__(self,mean,std,log_mean,classes=17,k=8,width=128):
  super().__init__();self.k=k;self.classes=classes;self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32));self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32))
  self.register_buffer('log_mean',torch.as_tensor(log_mean,dtype=torch.float32));self.input=BatchedLinear(k,len(mean),width)
  self.norm=nn.ModuleList([nn.LayerNorm(width) for i in range(3)]);self.up=nn.ModuleList([BatchedLinear(k,width,2*width) for i in range(3)]);self.down=nn.ModuleList([BatchedLinear(k,2*width,width) for i in range(3)])
  self.output=BatchedLinear(k,width,classes*3);self.final_norm=nn.LayerNorm(width);nn.init.normal_(self.output.w,std=.005)
  grid=torch.arange(60,1800,15,dtype=torch.float32);self.register_buffer('duration_grid',grid);self.register_buffer('log_edges',(grid[:-1]+7.5).log())
 def forward(self,x):
  h=F.silu(self.input(((x-self.mean)/self.std)[:,None].expand(-1,self.k,-1)))
  for norm,up,down in zip(self.norm,self.up,self.down):
   update=F.dropout(F.silu(up(norm(h))),p=.1,training=self.training);h=h+F.dropout(down(update),p=.1,training=self.training)
  raw=self.output(self.final_norm(h)).reshape(len(x),self.k,self.classes,3)
  loc=self.log_mean+2.5*torch.tanh(raw[...,0]/2.5);sigma=(.15+F.softplus(raw[...,1]-1.5)).clamp_max(1.5);weight=raw[...,2].log_softmax(-1)
  cdf=torch.special.ndtr((self.log_edges-loc[...,None])/sigma[...,None])
  component=torch.cat([cdf[...,:1],cdf[...,1:]-cdf[...,:-1],1-cdf[...,-1:]],-1).clamp_min(1e-7);component=component/component.sum(-1,keepdim=True);clp=component.log()
  marginal=torch.logsumexp(clp+weight[...,None],-2)
  return marginal,weight,clp
 def joint_auxiliary(self,out,code,target):
  lp,weight,conditional=out;ce=-weight.gather(-1,code[:,None,None].expand(-1,self.k,1)).mean()
  p=conditional.gather(-2,code[:,None,None,None].expand(-1,self.k,1,len(self.duration_grid)))[:,:,0]
  label=((target-60)/15).long();nll=-p.gather(-1,label[:,None,None].expand(-1,self.k,1)).mean()
  return .2*ce+.2*nll
 @torch.no_grad()
 def quantiles(self,lp):
  cdf=lp.exp().mean(1).cumsum(-1);q=torch.tensor([.1,.5,.9],device=lp.device);idx=(cdf[:,:,None]>=q).float().argmax(1)
  return self.duration_grid[idx]
