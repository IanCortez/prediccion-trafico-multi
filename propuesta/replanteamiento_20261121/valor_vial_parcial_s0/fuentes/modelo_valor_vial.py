"""Amortized street remaining-time model; public OD path never consumes route targets."""
import math,torch
from torch import nn
from torch.nn import functional as F
class RoadValueNet(nn.Module):
 def __init__(self,mean,std,static,neighbors,neighbor_valid):
  super().__init__();self.members=4;self.nroads=len(static);self.degree=neighbors.shape[1]
  for name,value in [('mean',mean),('std',std),('static',static)]:
   self.register_buffer(name,torch.as_tensor(value,dtype=torch.float32))
  self.static=F.pad(self.static,(0,0,0,1))
  self.register_buffer('neighbors',F.pad(torch.as_tensor(neighbors,dtype=torch.long),(0,0,0,1),value=self.nroads))
  self.register_buffer('neighbor_valid',F.pad(torch.as_tensor(neighbor_valid,dtype=torch.bool),(0,0,0,1)))
  self.embedding=nn.Embedding(self.nroads+1,48,padding_idx=self.nroads)
  nn.init.normal_(self.embedding.weight,std=.05)
  with torch.no_grad():self.embedding.weight[self.nroads].zero_()
  self.road_encoder=nn.Sequential(nn.Linear(108,96),nn.SiLU(),nn.Linear(96,64),nn.LayerNorm(64))
  self.global_encoder=nn.Sequential(nn.Linear(len(mean),384),nn.SiLU(),nn.LayerNorm(384),nn.Linear(384,256),nn.SiLU(),nn.LayerNorm(256))
  self.heads=nn.ModuleList([nn.Sequential(nn.Linear(392,128),nn.SiLU(),nn.Linear(128,7)) for k in range(4)])
  self.global_head=nn.Sequential(nn.Linear(256,64),nn.SiLU(),nn.Linear(64,28))
  self.local_head=nn.Sequential(nn.Linear(393,96),nn.SiLU(),nn.Linear(96,1))
  self.policy=nn.Sequential(nn.Linear(456,96),nn.SiLU(),nn.Linear(96,1))
  self.arrival=nn.Sequential(nn.Linear(392,64),nn.SiLU(),nn.Linear(64,1))
  for h in list(self.heads)+[self.global_head]:
   nn.init.normal_(h[-1].weight,std=.005);nn.init.zeros_(h[-1].bias)
  grid=torch.arange(60,1800,15,dtype=torch.float32);self.register_buffer('duration_grid',grid)
  self.register_buffer('log_edges',(grid[:-1]+7.5).log())
 def roads(self,ids):
  ids=ids.long();neighbor=self.neighbors[ids];valid=self.neighbor_valid[ids]
  pooled=(self.embedding(neighbor)*valid[...,None]).sum(-2)/valid.sum(-1,keepdim=True).clamp_min(1)
  return self.road_encoder(torch.cat([self.embedding(ids),pooled,self.static[ids]],-1))
 def hidden(self,x):return self.global_encoder((x-self.mean)/self.std)
 def context(self,hidden,origin,destination,of,df,oe,de,remaining):
  # All quantities here are public map state or query-derived context, never elapsed time.
  meta=torch.stack([of,df,oe/100,de/100,torch.log1p(remaining),remaining/10,
    torch.exp(self.static[origin.long(),8]*10).sub(1)/1000,
    torch.exp(self.static[destination.long(),8]*10).sub(1)/1000],-1)
  width=origin.shape[-1];h=hidden[:,None,:].expand(-1,width,-1)
  return torch.cat([h,self.roads(origin),self.roads(destination),meta],-1)
 def parameters_at(self,context,remaining):
  raw=torch.stack([h(context) for h in self.heads],1)
  base=(remaining*1000/8+15).log()[:,None,:,None]
  loc=base+.2+2*torch.tanh(raw[...,:2]/2)
  sigma=(.15+F.softplus(raw[...,2:4]-1.5)).clamp_max(1.5)
  weights=raw[...,4:6].log_softmax(-1)
  return loc,sigma,weights,raw[...,6]
 def probabilities(self,loc,sigma):
  cdf=torch.special.ndtr((self.log_edges-loc[...,None])/sigma[...,None])
  p=torch.cat([cdf[...,:1],cdf[...,1:]-cdf[...,:-1],1-cdf[...,-1:]],-1).clamp_min(1e-7)
  return p/p.sum(-1,keepdim=True)
 def public_logprob(self,x,chain,fraction,error_m,remaining_km,reachable,return_hidden=False):
  B=len(x);h=self.hidden(x);chain=chain.long();fraction=fraction.float();error_m=error_m.float();remaining=remaining_km.float().reshape(B,16)
  origin=chain[:,0,:,None].expand(-1,-1,4).reshape(B,16);destination=chain[:,1,None,:].expand(-1,4,-1).reshape(B,16)
  of=fraction[:,0,:,None].expand(-1,-1,4).reshape(B,16);df=fraction[:,1,None,:].expand(-1,4,-1).reshape(B,16)
  oe=error_m[:,0,:,None].expand(-1,-1,4).reshape(B,16);de=error_m[:,1,None,:].expand(-1,4,-1).reshape(B,16)
  context=self.context(h,origin,destination,of,df,oe,de,remaining)
  loc,sigma,style,score=self.parameters_at(context,remaining);p=self.probabilities(loc,sigma)
  valid=reachable.reshape(B,16).bool();exists=valid.any(1);safe=valid.clone();safe[:,0]|=~exists
  prior=-.5*((oe/30).square()+(de/30).square());score=(score+prior[:,None]).masked_fill(~safe[:,None],-1e9)
  pair=score.log_softmax(-1);raw=self.global_head(h).reshape(B,4,7)
  gloc=math.log(600)+2*torch.tanh(raw[...,:2]/2);gsigma=(.15+F.softplus(raw[...,2:4]-1.5)).clamp_max(1.5)
  gp=self.probabilities(gloc,gsigma);gw=raw[...,4:6].log_softmax(-1);gate=raw[...,6]
  road_gate=torch.where(exists[:,None],F.logsigmoid(gate),torch.full_like(gate,-torch.inf))
  global_gate=torch.where(exists[:,None],F.logsigmoid(-gate),torch.zeros_like(gate))
  rlog=torch.logsumexp((p.log()+style[...,None]+pair[:,:,:,None,None]).flatten(2,3),2)+road_gate[...,None]
  glog=torch.logsumexp(gp.log()+gw[...,None],2)+global_gate[...,None]
  logp=torch.logaddexp(rlog,glog)
  return (logp,h) if return_hidden else logp
 def auxiliary_loss(self,h,chain,destination,fraction,destination_fraction,remaining_km,remaining_time_s,local_time_s,local_distance_m,route_target,valid):
  valid=valid.bool()
  if not valid.any():return h.sum()*0,dict(value=0.,local=0.,route=0.)
  h=h[valid];chain=chain[valid].long();destination=destination[valid].long();fraction=fraction[valid].float();df=destination_fraction[valid].float()
  remaining=remaining_km[valid].float();zero=torch.zeros_like(remaining)
  context=self.context(h,chain[:,None],destination[:,None],fraction[:,None],df[:,None],zero[:,None],zero[:,None],remaining[:,None])
  loc,sigma,weights,score=self.parameters_at(context,remaining[:,None])
  target=remaining_time_s[valid].float().clamp_min(1).log()[:,None,None,None]
  logdensity=-.5*((target-loc)/sigma).square()-sigma.log()-target-.5*math.log(2*math.pi)
  value=-torch.logsumexp(logdensity+weights,-1).mean()
  distance=local_distance_m[valid].float();local_target=local_time_s[valid].float().clamp_min(0)
  local_raw=self.local_head(torch.cat([context[:,0],torch.log1p(distance)[:,None]/10],-1))[:,0]
  local_log=torch.log1p(distance/8+5)+2*torch.tanh(local_raw/2)
  local=F.smooth_l1_loss(local_log,torch.log1p(local_target))
  nextids=self.neighbors[chain];nextvalid=self.neighbor_valid[chain]
  nextroads=self.roads(nextids);expanded=context[:,0,None,:].expand(-1,self.degree,-1)
  logits=self.policy(torch.cat([expanded,nextroads],-1))[...,0].masked_fill(~nextvalid,-1e9)
  logits=torch.cat([logits,self.arrival(context[:,0])],-1);rt=route_target[valid].long();rv=rt>=0
  route=F.cross_entropy(logits[rv],rt[rv]) if rv.any() else logits.sum()*0
  return .05*value+.05*local+.1*route,dict(value=float(value.detach()),local=float(local.detach()),route=float(route.detach()))
 @torch.no_grad()
 def quantiles(self,logp):
  cdf=logp.exp().mean(1).cumsum(-1);q=torch.tensor([.1,.5,.9],device=logp.device)
  indices=(cdf[:,:,None]>=q).float().argmax(1)
  return self.duration_grid[indices]
