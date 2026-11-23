"""Historical34Dtokens withgeometry consistent withoriginalnormalizedarclengthteacher."""
from base_replanteo import *
import torch
from tokens_subtrayectos import SubtripTokenBank,query_parts_sub
class SubtripArcLengthBank(SubtripTokenBank):
 def __init__(self,device='cuda',shared=None):
  if shared is None:super().__init__(device)
  else:self.__dict__.update(shared.__dict__)
  meta=read(OUT/'distancia_arco_puntos_Train_s0/resultado.json')
  assert meta['complete'] and meta['source_point_bank_sha256']==read(OUT/'memoria_puntos_Train_s0/resultado.json')['outputs_sha256']
  file=OUT/'distancia_arco_puntos_Train_s0/distancia_acumulada_m.npz';assert sha(file)==meta['outputs_sha256']
  self.prefix_m=torch.as_tensor(np.load(file)['prefix_m'],device=self.xy.device)
  assert self.prefix_m.shape==(len(self.xy),) and int(self.n.max())<=120
 @torch.no_grad()
 def features(self,xy_m,hour,weekend,parent,first,last):
  device=xy_m.device
  parent=torch.as_tensor(parent,device=device).long();first=torch.as_tensor(first,device=device).long();last=torch.as_tensor(last,device=device).long()
  available=parent>=0;parent=parent.clamp_min(0)
  offset=self.offset[parent];start=self.xy[offset+first];end=self.xy[offset+last]
  source=torch.cat([start,end],-1);delta=(source-xy_m[:,None])/1000
  qd=xy_m[:,2:]-xy_m[:,:2];sd=end-start
  qnorm=qd.square().sum(-1).clamp_min(1e-12).sqrt();snorm=sd.square().sum(-1).clamp_min(1e-12).sqrt()
  scale=(qnorm[:,None]/snorm).clamp(.5,2)
  co=(qd[:,None]*sd).sum(-1)/(qnorm[:,None]*snorm)
  si=(qd[:,None,1]*sd[...,0]-qd[:,None,0]*sd[...,1])/(qnorm[:,None]*snorm)
  begin=self.prefix_m[offset+first];finish=self.prefix_m[offset+last]
  total_distance=(finish-begin).clamp_min(0)
  targets=begin[...,None]+total_distance[...,None]*self.fractions
  low=first[...,None].expand(-1,-1,7).clone();high=last[...,None].expand(-1,-1,7).clone()
  # At most120sourceGPSpoints; seven lowerbound iterations suffice, including repeatedstationarypoints.
  for iteration in range(7):
   mid=(low+high)//2;below=self.prefix_m[offset[...,None]+mid]<targets
   low=torch.where(below,mid+1,low);high=torch.where(below,high,mid)
  hi=low;lo=torch.maximum(hi-1,first[...,None])
  left=self.prefix_m[offset[...,None]+lo];right=self.prefix_m[offset[...,None]+hi]
  f=((targets-left)/(right-left).clamp_min(1e-9)).clamp(0,1)
  middle=self.xy[offset[...,None]+lo]*(1-f[...,None])+self.xy[offset[...,None]+hi]*f[...,None]
  residual=middle-start[...,None,:]-self.fractions[None,None,:,None]*sd[...,None,:]
  a=residual[...,0];b=residual[...,1]
  shape=torch.stack([scale[...,None]*(co[...,None]*a-si[...,None]*b),
                     scale[...,None]*(si[...,None]*a+co[...,None]*b)],-1).flatten(-2)/1000
  path=total_distance*scale/1000
  circuity=(path/(qnorm[:,None]/1000)).clamp_min(.01).log()
  duration=(15*(last-first)*scale).clamp_min(.1).log()
  distance=delta.square().sum(-1).sqrt()
  ts=self.ts[parent]+15*first;offset_s=torch.zeros_like(ts)
  for k in range(len(self.starts)):offset_s=torch.where((ts>=self.starts[k])&(ts<self.ends[k]),3600,offset_s)
  local=ts+offset_s;source_hour=(local%86400).float()/3600;source_weekend=((local//86400+3)%7)>=5
  clock=torch.remainder(source_hour-hour[:,None]+12,24)-12;phase=clock*(2*np.pi/24)
  match=(source_weekend==weekend[:,None]).float()
  prior=-.5*(distance/.05).square()-.5*(clock/4).square()+torch.where(match>.5,0.,float(np.log(.35)))
  den=(self.n[parent]-1).clamp_min(1).float()
  context=torch.stack([first/den,(self.n[parent]-1-last)/den,(first==0).float(),
                       (last==self.n[parent]-1).float(),(den*15).log(),(last-first)/den],-1)
  raw=torch.cat([shape,path[...,None],circuity[...,None],duration[...,None],delta,distance[...,None],
    phase.sin()[...,None],phase.cos()[...,None],match[...,None],scale.log()[...,None],prior[...,None],available.float()[...,None],context],-1)
  assert raw.shape==(len(xy_m),64,34)
  return torch.where(available[...,None],raw,0.).to(torch.float16)
