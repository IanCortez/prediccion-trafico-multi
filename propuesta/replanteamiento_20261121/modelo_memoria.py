"""Learned historical-label kernel, inspired by ModernNCA. Static query features only."""
import torch,math
from torch import nn

class LearnedMemory(nn.Module):
 def __init__(self,n_features):
  super().__init__();self.encoder=nn.Sequential(nn.Linear(n_features,256),nn.ReLU(),nn.LayerNorm(256),nn.Dropout(.05),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,32));self.log_tau=nn.Parameter(torch.tensor(math.log(.05)));self.log_geo_bw=nn.Parameter(torch.zeros(2))
 def pack(self,x,geo):
  z=nn.functional.normalize(self.encoder(x),dim=-1);tau=self.log_tau.exp().clamp(.001,2);bw=self.log_geo_bw.exp().clamp(.05,10);return torch.cat([z/tau.sqrt(),geo[:,:2]/bw[0],geo[:,2:]/bw[1]],-1)
 @staticmethod
 def scores(q,k):
  return -(q.square().sum(-1,keepdim=True)+k.square().sum(-1)[None,:]-2*q@k.T).clamp_min(0)
 def neighborhood(self,xq,gq,xc,gc,query_ids,candidate_ids):
  scores=self.scores(self.pack(xq,gq),self.pack(xc,gc));scores=scores.masked_fill(query_ids[:,None]==candidate_ids[None,:],-torch.inf);return scores.softmax(-1)

@torch.no_grad()
def stream_distribution(q,keys,onehot,prior,chunk=65536):
 # Exact soft-neighbor aggregation over ALL keys with a stable streaming normalizer.
 acc=torch.zeros((len(q),116),device=q.device);mass=torch.zeros(len(q),device=q.device);peak=torch.full((len(q),),-torch.inf,device=q.device)
 for first in range(0,len(keys),chunk):
  score=LearnedMemory.scores(q,keys[first:first+chunk]);newpeak=torch.maximum(peak,score.max(-1).values);oldscale=(peak-newpeak).exp();e=(score-newpeak[:,None]).exp();acc=acc*oldscale[:,None]+e@onehot[first:first+chunk];mass=mass*oldscale+e.sum(-1);peak=newpeak
 p=acc/mass[:,None].clamp_min(1e-30);return .99*p+.01*prior
