"""Matchedquerygate with stablelogdensity mixture and exactunsupportedreference."""
import torch,math
from torch import nn
import torch.nn.functional as F
class RoadCDFGate(nn.Module):
 def __init__(self,mean,std):
  super().__init__();self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32));self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32))
  self.net=nn.Sequential(nn.Linear(206,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,1))
  nn.init.zeros_(self.net[-1].weight);nn.init.constant_(self.net[-1].bias,math.log(.05/.95))
 def forward(self,x):
  return self.net(((x-self.mean)/self.std).clamp(-15,15)).squeeze(-1)
def mixture_logprob(logit,reference_logprob,graph,supported):
 logit=logit.double()
 graph_logprob=graph.double().log()
 mixed=torch.logaddexp(F.logsigmoid(-logit)[:,None]+reference_logprob,F.logsigmoid(logit)[:,None]+graph_logprob)
 return torch.where(supported[:,None],mixed,reference_logprob)
@torch.inference_mode()
def canonical_gate(model,x,reference_logprob,graph,supported):
 model.eval();logs=[];gates=[]
 for begin in range(0,len(x),256):
  n=min(256,len(x)-begin);xx=torch.zeros((256,206),dtype=x.dtype,device=x.device);xx[:n]=x[begin:begin+n]
  logits=model(xx)[:n].double()
  logs.append(mixture_logprob(logits,reference_logprob[begin:begin+n],graph[begin:begin+n],supported[begin:begin+n]).cpu())
  gates.append(logits.sigmoid().cpu())
 return torch.cat(logs),torch.cat(gates)
def point_quantiles(logp):
 cdf=logp.exp().cumsum(-1);levels=torch.tensor([.1,.5,.9],dtype=cdf.dtype,device=cdf.device)
 index=(cdf[...,None]>=levels).to(torch.int64).argmax(1)
 return (index+4)*15
