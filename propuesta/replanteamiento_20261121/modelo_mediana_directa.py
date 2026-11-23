"""Stable explicit MAE readout anchored to an immutable certified predictor."""
import copy,torch
from torch import nn
class DirectMedianReadout(nn.Module):
 def __init__(self,reference,train_student=False):
  super().__init__();self.reference=reference
  for p in self.reference.parameters():p.requires_grad_(False)
  self.student=copy.deepcopy(reference);self.train_student=False
  self.readout=nn.Sequential(nn.LayerNorm(578),nn.Linear(578,128),nn.SiLU(),nn.Linear(128,1))
  nn.init.zeros_(self.readout[-1].weight);nn.init.zeros_(self.readout[-1].bias)
  self.set_student_training(train_student)
 @property
 def grid(self):return self.reference.duration.duration_grid
 def set_student_training(self,enabled):
  self.train_student=bool(enabled)
  for p in self.student.parameters():p.requires_grad_(enabled)
  self.student.train(self.training and enabled);self.reference.eval()
 def train(self,mode=True):
  super().train(mode);self.reference.eval();self.student.train(mode and self.train_student)
  return self
 @torch.no_grad()
 def reference_point(self,x,public,history):
  probability=self.reference(x,public,history)[0].exp().mean(1)
  return self.grid[(probability.cumsum(-1)<.5).sum(-1).clamp_max(115)]
 def compute(self,x,public,history,frozen_reference_prediction):
  assert frozen_reference_prediction.shape==(len(x),) and not frozen_reference_prediction.requires_grad
  hidden=[];hook=self.student.duration.backbone.output.register_forward_pre_hook(lambda module,args:hidden.append(args[0]))
  try:
   if self.train_student:out=self.student(x,public,history)
   else:
    with torch.no_grad():out=self.student(x,public,history)
  finally:hook.remove()
  assert len(hidden)==1
  moments=[];grid=self.grid/600
  for first in range(0,len(x),256):
   p=out[4][first:first+256].exp()
   mean=(p*grid).sum(-1)
   sd=(p*(grid-mean[...,None]).square()).sum(-1).clamp_min(1e-8).sqrt()
   moments.append(torch.cat([out[1][first:first+256].exp(),mean,sd],-1))
  context=torch.cat([hidden[0],torch.cat(moments,0)],-1)
  assert context.shape[-1]==578 and torch.isfinite(context).all()
  raw=self.readout(context).squeeze(-1)
  members=(frozen_reference_prediction[:,None]+600*torch.tanh(raw)).clamp(60,1785)
  return members,out
 def forward(self,x,public,history):
  point=self.reference_point(x,public,history)
  return self.compute(x,public,history,point)[0]
 def training_forward(self,x,public,history,frozen_reference_prediction):
  assert self.training
  return self.compute(x,public,history,frozen_reference_prediction)
 @staticmethod
 def quantize(point):return (60+15*torch.floor((point-60)/15+.5)).clamp(60,1785)
