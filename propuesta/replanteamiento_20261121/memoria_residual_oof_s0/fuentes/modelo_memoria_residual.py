"""Cross-fitted residual kernel with MAE supervision and explicit base fallback."""
import torch,math
from torch import nn
class ResidualMemory(nn.Module):
 def __init__(self,n_features):
  super().__init__()
  self.encoder=nn.Sequential(nn.Linear(n_features,256),nn.ReLU(),nn.LayerNorm(256),nn.Dropout(.05),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,32))
  self.gate=nn.Linear(32,1);nn.init.zeros_(self.gate.weight);nn.init.constant_(self.gate.bias,-1.)
  self.log_tau=nn.Parameter(torch.tensor(math.log(.05)));self.log_geo_bw=nn.Parameter(torch.zeros(2))
 def encode(self,x,geo):
  h=self.encoder(x);z=nn.functional.normalize(h,dim=-1);tau=self.log_tau.exp().clamp(.001,2);bw=self.log_geo_bw.exp().clamp(.05,10)
  return torch.cat([z/tau.sqrt(),geo[:,:2]/bw[0],geo[:,2:]/bw[1]],-1),self.gate(h).sigmoid()[:,0]
 @staticmethod
 def scores(q,k):return -(q.square().sum(-1,keepdim=True)+k.square().sum(-1)[None,:]-2*q@k.T).clamp_min(0)
 def correction(self,xq,gq,xc,gc,qids,cids,values):
  q,gate=self.encode(xq,gq);k,_=self.encode(xc,gc);s=self.scores(q,k).masked_fill(qids[:,None]==cids[None,:],-torch.inf);w=s.softmax(-1)
  return gate*.99*(w@values),w
@torch.no_grad()
def stream_value(q,keys,values,chunk=65536):
 acc=torch.zeros(len(q),device=q.device);mass=torch.zeros_like(acc);peak=torch.full_like(acc,-torch.inf)
 for first in range(0,len(keys),chunk):
  s=ResidualMemory.scores(q,keys[first:first+chunk]);new=torch.maximum(peak,s.max(-1).values);scale=(peak-new).exp();e=(s-new[:,None]).exp();acc=acc*scale+e@values[first:first+chunk];mass=mass*scale+e.sum(-1);peak=new
 return acc/mass.clamp_min(1e-30)

