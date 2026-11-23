"""Attention over two ordered public-map routes; five-field query information only."""
import torch
from torch import nn
class PublicRouteAttention(nn.Module):
 def __init__(self,duration,road_static,road_confidence,clock_indices,mode='route'):
  super().__init__();self.duration=duration;self.mode=mode
  self.register_buffer('road_static',torch.as_tensor(road_static,dtype=torch.float32))
  self.register_buffer('road_confidence',torch.as_tensor(road_confidence,dtype=torch.float32))
  self.register_buffer('clock_indices',torch.as_tensor(clock_indices,dtype=torch.long))
  self.edge_embedding=nn.Embedding(len(road_static),16,padding_idx=0)
  nn.init.normal_(self.edge_embedding.weight,std=.01)
  with torch.no_grad():self.edge_embedding.weight[0].zero_()
  self.input=nn.Linear(30,64)
  self.position=nn.Parameter(torch.randn(1,32,64)*.01)
  self.route_type=nn.Parameter(torch.randn(2,1,64)*.01)
  self.cls=nn.Parameter(torch.randn(1,1,64)*.01)
  layer=nn.TransformerEncoderLayer(64,4,128,.1,batch_first=True,norm_first=True,activation='gelu')
  self.encoder=nn.TransformerEncoder(layer,2,norm=nn.LayerNorm(64),enable_nested_tensor=False)
  self.readout=nn.Sequential(nn.Linear(136,128),nn.SiLU(),nn.Linear(128,24))
  nn.init.zeros_(self.readout[-1].weight);nn.init.zeros_(self.readout[-1].bias)
 def branch_parameters(self):
  return [p for n,p in self.named_parameters() if not n.startswith('duration.')]
 def route_delta(self,x,tokens):
  assert tokens.shape==(len(x),2,32)
  if self.mode=='clock_control':tokens=torch.zeros_like(tokens)
  clock=x[:,self.clock_indices]
  embed=self.edge_embedding(tokens)*self.road_confidence[tokens][...,None]
  static=self.road_static[tokens];context=clock[:,None,None,:].expand(-1,2,32,-1)
  z=self.input(torch.cat([embed,static,context],-1)).reshape(len(x)*2,32,64)
  z=z+self.position+self.route_type[None].expand(len(x),-1,-1,-1).reshape(len(x)*2,1,64)
  c=self.cls.expand(len(x)*2,-1,-1);z=torch.cat([c,z],1)
  padding=torch.cat([torch.zeros((len(x)*2,1),device=x.device,dtype=torch.bool),(tokens==0).reshape(len(x)*2,32)],1)
  encoded=self.encoder(z,src_key_padding_mask=padding)[:,0].reshape(len(x),128)
  return self.readout(torch.cat([encoded,clock],-1)).reshape(len(x),8,3)
 def forward(self,x,tokens):
  delta=self.route_delta(x,tokens)
  def add_delta(module,args,out):
   raw=out.reshape(len(x),-1,8,31)
   return torch.cat([raw[...,:3]+delta[:,None],raw[...,3:]],-1).reshape_as(out)
  hook=self.duration.backbone.output.register_forward_hook(add_delta)
  try:return self.duration(x)
  finally:hook.remove()
