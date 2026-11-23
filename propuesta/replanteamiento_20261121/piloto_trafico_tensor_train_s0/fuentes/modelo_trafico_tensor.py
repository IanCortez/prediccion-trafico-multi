"""Low-rank road/weekquarterhour pace+dwell, fitted independently in each source-excluded view."""
import torch
from torch import nn
from torch.nn import functional as F
class RoadTrafficTensor(nn.Module):
 def __init__(self,static,basepace,support):
  super().__init__();static=torch.as_tensor(static,dtype=torch.float32)
  self.register_buffer('static',(static-static.mean(0))/static.std(0).clamp_min(.01));self.register_buffer('basepace',torch.as_tensor(basepace,dtype=torch.float32))
  self.register_buffer('support',torch.as_tensor(support,dtype=torch.float32))
  n=len(static);self.road=nn.Embedding(n,32);self.clock=nn.Embedding(672,32);self.road_bias=nn.Embedding(n,2);self.clock_bias=nn.Embedding(672,2)
  self.background=nn.Sequential(nn.Linear(12,32),nn.SiLU(),nn.Linear(32,2))
  nn.init.normal_(self.road.weight,std=.05);nn.init.normal_(self.clock.weight,std=.05);nn.init.zeros_(self.road_bias.weight);nn.init.zeros_(self.clock_bias.weight)
  nn.init.zeros_(self.background[-1].weight);nn.init.zeros_(self.background[-1].bias)
 def components(self,road,slot):
  road=road.long();slot=slot.long();shrink=(self.support[road]/(self.support[road]+10))[...,None]
  latent=(self.road(road).reshape(-1,2,16)*self.clock(slot).reshape(-1,2,16)).sum(-1)/4
  score=self.background(self.static[road])+self.clock_bias(slot)+(latent+self.road_bias(road))*shrink
  pace=self.basepace[road]*torch.exp(2*torch.tanh(score[...,0]/2));dwell=F.softplus(score[...,1]+2)
  return pace,dwell
 def forward(self,road,slot,distance):
  pace,dwell=self.components(road,slot);return pace*distance+dwell
 def smoothness(self):
  return (self.clock.weight-self.clock.weight.roll(1,0)).square().mean()+(self.clock_bias.weight-self.clock_bias.weight.roll(1,0)).square().mean()
