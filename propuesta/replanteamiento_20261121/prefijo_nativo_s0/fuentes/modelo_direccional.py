"""Spatial semi-Markov state=(cell,incoming cardinal direction), unknown direction only at origin."""
import torch
from torch import nn
from modelo_espacial import SpatialArrival
from nucleo_direccional import directional_arrival
def heading_ids(dx,dy):
 return torch.where(dx.abs()>=dy.abs(),torch.where(dx>=0,0,1),torch.where(dy>=0,2,3)).long()
class DirectionalArrival(SpatialArrival):
 def __init__(self,stats,grid,graph,distance,power):
  super().__init__(stats,grid,graph,distance,power)
  self.heading=nn.Embedding(5,8);self.state_encoder=nn.Sequential(nn.Linear(64+32+7+8,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU());self.end_direction=nn.Linear(8,121,bias=False);nn.init.zeros_(self.end_direction.weight)
  dx=torch.as_tensor(graph['offsets'][:,0],dtype=torch.long);dy=torch.as_tensor(graph['offsets'][:,1],dtype=torch.long);h=heading_ids(dx,dy);self.register_buffer('outgoing_heading',h,persistent=False);nb=torch.where(self.neighbors>=0,self.neighbors*4+h[None,:],-1);self.register_buffer('directed_neighbors',nb.repeat_interleave(4,dim=0),persistent=False)
 def state_parameters(self,q,geo,cell_ids,destination,heading):
  current=self.centers[cell_ids];dest=self.centers[destination];d=self.distance[cell_ids,destination];fallback=((current-dest).square().sum(-1).sqrt()*1.5);d=torch.where(torch.isfinite(d),d,fallback);rel=torch.cat([current/5000,(current-geo[:,:2]*1000)/5000,(current-geo[:,2:4]*1000)/5000,d[:,None]/5000],-1);h=self.state_encoder(torch.cat([q,self.cell(cell_ids),rel,self.heading(heading)],-1));nb=self.neighbors[cell_ids];nc=self.centers[nb.clamp_min(0)];nd=self.distance[nb.clamp_min(0),destination[:,None]];nf=((nc-dest[:,None]).square().sum(-1).sqrt()*1.5);nd=torch.where(torch.isfinite(nd),nd,nf);progress=((nd-d[:,None])/1000).clamp(-2,2);logits=self.route_head(h).float()-nn.functional.softplus(self.progress_log_scale)*progress;mask=self.allowed[cell_ids];p=logits.masked_fill(~mask,-1e9).softmax(-1)*mask;wait=(self.wait_head(h).float()+self.move_prior).softmax(-1);return p,wait
 def end_parameters(self,e,we,heading):
  delta=self.end_direction(self.heading(heading)).float();ee=(torch.logit(e.clamp(1e-6,1-1e-6))+delta[:,0]).sigmoid();ww=(we.clamp_min(1e-30).log()+delta[:,1:]).softmax(-1);return ee,ww
 def process_parameters(self,x,origin=None,destination=None):
  q,o,d,p0,e,we=self.encode_od(x,origin,destination);B=len(x);S=self.N*4+1;cell=torch.arange(self.N,device=x.device).repeat_interleave(4);cells=torch.cat([cell[None,:].expand(B,-1),o[:,None]],1).reshape(-1);heading=torch.cat([torch.arange(4,device=x.device).repeat(self.N)[None,:].expand(B,-1),torch.full((B,1),4,device=x.device)],1).reshape(-1);qq=q[:,None,:].expand(B,S,64).reshape(-1,64);geo=x[:,:4][:,None,:].expand(B,S,4).reshape(-1,4);dd=d[:,None].expand(B,S).reshape(-1);pol=[];wait=[]
  for first in range(0,len(cells),65536):
   a,b=self.state_parameters(qq[first:first+65536],geo[first:first+65536],cells[first:first+65536],dd[first:first+65536],heading[first:first+65536]);pol.append(a);wait.append(b)
  hh=torch.arange(5,device=x.device)[None,:].expand(B,-1).reshape(-1);ee,ww=self.end_parameters(e[:,None].expand(B,5).reshape(-1),we[:,None,:].expand(B,5,120).reshape(-1,120),hh);ee=ee.reshape(B,5);ee=ee*torch.cat([torch.ones(B,4,device=x.device),(o==d).float()[:,None]],1);ww=ww.reshape(B,5,120)
  initial_nb=torch.where(self.neighbors[o]>=0,self.neighbors[o]*4+self.outgoing_heading[None,:],-1);neighbors=torch.cat([self.directed_neighbors[None,:,:].expand(B,-1,-1),initial_nb[:,None,:]],1);initial=torch.full_like(o,S-1);end_states=torch.cat([d[:,None]*4+torch.arange(4,device=x.device)[None,:],torch.full((B,1),S-1,device=x.device,dtype=torch.long)],1)
  return torch.cat(pol).reshape(B,S,48),torch.cat(wait).reshape(B,S,119),ww,ee,neighbors,initial,end_states,p0
 def posterior(self,x,origin=None,destination=None):
  policy,wait,we,e,neighbors,initial,ends,p0=self.process_parameters(x,origin,destination);raw=directional_arrival(policy,wait,we,e,neighbors,initial,ends);mass=raw[:,4:120].sum(-1);path=raw[:,4:120]/mass[:,None].clamp_min(1e-12);alpha=self.mix_logit.sigmoid()*mass/(mass+.01);p=(1-alpha[:,None])*p0+alpha[:,None]*path;p=p/p.sum(-1,keepdim=True);return p,mass,p0,path

