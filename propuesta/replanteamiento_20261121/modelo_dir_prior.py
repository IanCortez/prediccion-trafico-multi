"""Hierarchical empirical transition prior plus neural correction; Train prior excludes query fold."""
import torch,math,numpy as np
from torch import nn
from modelo_direccional import DirectionalArrival
from nucleo_direccional import directional_arrival
class PriorDirectionalArrival(DirectionalArrival):
 def __init__(self,stats,grid,graph,distance,power,counts):
  super().__init__(stats,grid,graph,distance,power)
  assert counts.shape==(5,self.N*32*5,48)
  fold=torch.as_tensor(counts.astype(np.float32));total=fold.sum(0);base=total.reshape(self.N*32,5,48).sum(1);basefold=fold.reshape(5,self.N*32,5,48).sum(2);glob=total.reshape(self.N,32,5,48).sum((0,2));globfold=fold.reshape(5,self.N,32,5,48).sum((1,3))
  for name,value in [('counts_fold',fold),('counts_total',total),('counts_base',base),('counts_base_fold',basefold),('counts_global',glob),('counts_global_fold',globfold)]:self.register_buffer(name,value,persistent=False)
  self.prior_log_strength=nn.Parameter(torch.tensor(math.log(math.exp(1)-1)))
 def prior_probabilities(self,geo,cell_ids,heading,excluded_fold=None):
  rel=geo[:,2:4]*1000-self.centers[cell_ids];angle=torch.remainder(torch.floor((torch.atan2(rel[:,1],rel[:,0])+math.pi/8)/(math.pi/4)).long(),8);distance=(rel.square().sum(-1).sqrt()[:,None]>torch.tensor([500.,1500.,4000.],device=geo.device)[None,:]).sum(-1);co=angle*4+distance;g=cell_ids*32+co;h=g*5+heading;ch=self.counts_total[h];cb=self.counts_base[g];cg=self.counts_global[co]
  if excluded_fold is not None:ch=ch-self.counts_fold[excluded_fold,h];cb=cb-self.counts_base_fold[excluded_fold,g];cg=cg-self.counts_global_fold[excluded_fold,co]
  gp=(cg+1)/(cg.sum(-1,keepdim=True)+48);p0=(cb+20*gp)/(cb.sum(-1,keepdim=True)+20);p1=(ch+20*p0)/(ch.sum(-1,keepdim=True)+20);return p1
 def state_parameters(self,q,geo,cell_ids,destination,heading,excluded_fold=None):
  p,wait=super().state_parameters(q,geo,cell_ids,destination,heading);prior=self.prior_probabilities(geo,cell_ids,heading,excluded_fold);strength=nn.functional.softplus(self.prior_log_strength).clamp_max(3);logits=p.clamp_min(1e-30).log()+strength*prior.clamp_min(1e-30).log();mask=self.allowed[cell_ids];p=logits.masked_fill(~mask,-1e9).softmax(-1)*mask;return p,wait
 def process_parameters(self,x,origin=None,destination=None,excluded_fold=None):
  q,o,d,p0,e,we=self.encode_od(x,origin,destination);B=len(x);S=self.N*4+1;cell=torch.arange(self.N,device=x.device).repeat_interleave(4);cells=torch.cat([cell[None,:].expand(B,-1),o[:,None]],1).reshape(-1);heading=torch.cat([torch.arange(4,device=x.device).repeat(self.N)[None,:].expand(B,-1),torch.full((B,1),4,device=x.device)],1).reshape(-1);qq=q[:,None,:].expand(B,S,64).reshape(-1,64);geo=x[:,:4][:,None,:].expand(B,S,4).reshape(-1,4);dd=d[:,None].expand(B,S).reshape(-1);ff=None if excluded_fold is None else excluded_fold[:,None].expand(B,S).reshape(-1);pol=[];wait=[]
  for first in range(0,len(cells),32768):
   last=first+32768;a,b=self.state_parameters(qq[first:last],geo[first:last],cells[first:last],dd[first:last],heading[first:last],None if ff is None else ff[first:last]);pol.append(a);wait.append(b)
  hh=torch.arange(5,device=x.device)[None,:].expand(B,-1).reshape(-1);ee,ww=self.end_parameters(e[:,None].expand(B,5).reshape(-1),we[:,None,:].expand(B,5,120).reshape(-1,120),hh);ee=ee.reshape(B,5)*torch.cat([torch.ones(B,4,device=x.device),(o==d).float()[:,None]],1);ww=ww.reshape(B,5,120);initial_nb=torch.where(self.neighbors[o]>=0,self.neighbors[o]*4+self.outgoing_heading[None,:],-1);neighbors=torch.cat([self.directed_neighbors[None,:,:].expand(B,-1,-1),initial_nb[:,None,:]],1);initial=torch.full_like(o,S-1);end_states=torch.cat([d[:,None]*4+torch.arange(4,device=x.device)[None,:],torch.full((B,1),S-1,device=x.device,dtype=torch.long)],1)
  return torch.cat(pol).reshape(B,S,48),torch.cat(wait).reshape(B,S,119),ww,ee,neighbors,initial,end_states,p0
 def posterior(self,x,origin=None,destination=None,excluded_fold=None):
  policy,wait,we,e,neighbors,initial,ends,p0=self.process_parameters(x,origin,destination,excluded_fold);raw=directional_arrival(policy,wait,we,e,neighbors,initial,ends);mass=raw[:,4:120].sum(-1);path=raw[:,4:120]/mass[:,None].clamp_min(1e-12);alpha=self.mix_logit.sigmoid()*mass/(mass+.01);p=(1-alpha[:,None])*p0+alpha[:,None]*path;p=p/p.sum(-1,keepdim=True);return p,mass,p0,path

