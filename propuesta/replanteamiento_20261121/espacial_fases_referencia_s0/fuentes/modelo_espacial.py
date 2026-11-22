"""OD-conditioned semi-Markov spatial process. Unknown query paths are marginalized."""
import torch,math
from torch import nn
import rtdl_num_embeddings as rtdl
from nucleo_semi_markov import arrival_distribution

class SpatialArrival(nn.Module):
 def __init__(self,stats,grid,graph,distance,power):
  super().__init__();self.N=len(graph['centers']);self.grid=grid;self.H=120
  for name,value,dtype in [('mean',stats['mean'],torch.float32),('std',stats['std'],torch.float32),('keep',stats['keep'],torch.long),('power',power,torch.float32)]:self.register_buffer(name,torch.as_tensor(value,dtype=dtype))
  for name,value,dtype in [('centers',graph['centers'],torch.float32),('neighbors',graph['neighbors'],torch.long),('allowed',graph['allowed'],torch.bool),('distance',distance,torch.float32)]:self.register_buffer(name,torch.as_tensor(value,dtype=dtype),persistent=False)
  self.register_buffer('duration_grid',torch.arange(60,1786,15,dtype=torch.float32));self.cell=nn.Embedding(self.N+1,32);self.numeric=rtdl.PiecewiseLinearEmbeddings(stats['bins'],d_embedding=16,activation=False,version='B');self.od_encoder=nn.Sequential(nn.Linear(len(stats['keep'])*16+64,512),nn.ReLU(),nn.Dropout(.1),nn.Linear(512,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU());self.od_head=nn.Linear(64,116);nn.init.zeros_(self.od_head.weight);nn.init.zeros_(self.od_head.bias);self.state_encoder=nn.Sequential(nn.Linear(64+32+7,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU());self.route_head=nn.Linear(64,48);self.wait_head=nn.Linear(64,119);self.end_head=nn.Linear(64,1);self.end_wait_head=nn.Linear(64,120);self.progress_log_scale=nn.Parameter(torch.tensor(math.log(math.exp(2)-1)));self.mix_logit=nn.Parameter(torch.tensor(0.))
  counts=torch.arange(119,dtype=torch.float32);p=.99*.5*torch.pow(.5,counts);p=p/p.sum()*.99+.01/119;self.register_buffer('move_prior',p.log());counts=torch.arange(120,dtype=torch.float32);p=.49*.5*torch.pow(.5,counts);p=p/p.sum()*.49+.01/120;p[0]+=.5;self.register_buffer('end_prior',p.log())
 def grid_ids(self,xy_km):
  g=self.grid;a=torch.floor((xy_km[:,0]*1000-g['x0'])/g['cell_m']).long();b=torch.floor((xy_km[:,1]*1000-g['y0'])/g['cell_m']).long();return a.clamp(0,g['nx']-1)*g['ny']+b.clamp(0,g['ny']-1)
 def encode_od(self,x,origin=None,destination=None):
  if origin is None:origin=self.grid_ids(x[:,:2])
  if destination is None:destination=self.grid_ids(x[:,2:4])
  z=(x[:,self.keep]-self.mean)/self.std;num=self.numeric(z).flatten(1);q=self.od_encoder(torch.cat([num,self.cell(origin),self.cell(destination)],-1));distance=(x[:,6]*1000).clamp_min(1);loc=self.power[0]+self.power[1]*distance.log();lg=self.duration_grid.log();prior=-.5*((lg[None,:]-loc[:,None])/.6).square()-lg;od=(self.od_head(q).float()+prior).softmax(-1);end=self.end_head(q).float().sigmoid()[:,0];end_wait=(self.end_wait_head(q).float()+self.end_prior).softmax(-1)
  return q,origin,destination,od,end,end_wait
 def state_parameters(self,q,geo,cell_ids,destination):
  current=self.centers[cell_ids];dest=self.centers[destination];d=self.distance[cell_ids,destination];fallback=((current-dest).square().sum(-1).sqrt()*1.5);d=torch.where(torch.isfinite(d),d,fallback);rel=torch.cat([current/5000,(current-geo[:,:2]*1000)/5000,(current-geo[:,2:4]*1000)/5000,d[:,None]/5000],-1);h=self.state_encoder(torch.cat([q,self.cell(cell_ids),rel],-1));nb=self.neighbors[cell_ids];nc=self.centers[nb.clamp_min(0)];nd=self.distance[nb.clamp_min(0),destination[:,None]];nf=((nc-dest[:,None]).square().sum(-1).sqrt()*1.5);nd=torch.where(torch.isfinite(nd),nd,nf);progress=((nd-d[:,None])/1000).clamp(-2,2);logits=self.route_head(h).float()-nn.functional.softplus(self.progress_log_scale)*progress;mask=self.allowed[cell_ids];p=logits.masked_fill(~mask,-1e9).softmax(-1)*mask;wait=(self.wait_head(h).float()+self.move_prior).softmax(-1);return p,wait
 def process_parameters(self,x,origin=None,destination=None):
  q,o,d,p0,e,we=self.encode_od(x,origin,destination);B=len(x);cells=torch.arange(self.N,device=x.device).repeat(B);qq=q[:,None,:].expand(B,self.N,64).reshape(-1,64);geo=x[:,:4,None].transpose(1,2).expand(B,self.N,4).reshape(-1,4);dd=d[:,None].expand(B,self.N).reshape(-1);pol=[];wait=[]
  for first in range(0,len(cells),65536):
   a,b=self.state_parameters(qq[first:first+65536],geo[first:first+65536],cells[first:first+65536],dd[first:first+65536]);pol.append(a);wait.append(b)
  return torch.cat(pol).reshape(B,self.N,48),torch.cat(wait).reshape(B,self.N,119),we,e,o,d,p0
 def posterior(self,x,origin=None,destination=None):
  policy,wait,we,e,o,d,p0=self.process_parameters(x,origin,destination);raw=arrival_distribution(policy,wait,we,e,self.neighbors,o,d);mass=raw[:,4:120].sum(-1);path=raw[:,4:120]/mass[:,None].clamp_min(1e-12);alpha=self.mix_logit.sigmoid()*mass/(mass+.01);p=(1-alpha[:,None])*p0+alpha[:,None]*path;p=p/p.sum(-1,keepdim=True);return p,mass,p0,path
 def median(self,p):return self.duration_grid[(p.cumsum(-1)<.5).sum(-1).clamp_max(115)]
