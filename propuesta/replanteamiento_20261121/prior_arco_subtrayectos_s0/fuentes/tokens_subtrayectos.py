"""34D historical-subtrajectory features from bounded source indices, never query path."""
from base_replanteo import *
import torch,pandas as pd
from recuperar_subtrayectos import metric_xy
from datos_progreso_recurrente import lisbon_transitions
def query_parts_sub(d):
 local=pd.to_datetime(d['timestamp'],unit='s',utc=True).tz_convert('Europe/Lisbon')
 return dict(xy_m=metric_xy(d).astype(np.float32),hour=(np.asarray(local.hour)+np.asarray(local.minute)/60+np.asarray(local.second)/3600).astype(np.float32),
  weekend=np.asarray(local.dayofweek)>=5)
class SubtripTokenBank:
 def __init__(self,device='cuda'):
  folder=OUT/'memoria_puntos_Train_s0';meta=read(folder/'resultado.json')
  assert sha(folder/'puntos_Train.npz')==meta['outputs_sha256'];z=dict(np.load(folder/'puntos_Train.npz'))
  self.xy=torch.as_tensor(z['xy_m'],device=device)
  self.offset=torch.as_tensor(z['offsets'],device=device)
  self.n=torch.as_tensor(z['parent_gps_n'],device=device,dtype=torch.long)
  self.ts=torch.as_tensor(z['timestamp'],device=device,dtype=torch.long)
  starts,ends=lisbon_transitions()
  self.starts=torch.as_tensor(starts,device=device);self.ends=torch.as_tensor(ends,device=device)
  self.fractions=torch.arange(1,8,device=device,dtype=torch.float32)/8
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
  pos=first[...,None]+(last-first)[...,None]*self.fractions
  lo=pos.long();hi=torch.minimum(lo+1,last[...,None]);f=pos-lo
  middle=self.xy[offset[...,None]+lo]*(1-f[...,None])+self.xy[offset[...,None]+hi]*f[...,None]
  residual=middle-start[...,None,:]-self.fractions[None,None,:,None]*sd[...,None,:]
  a=residual[...,0];b=residual[...,1]
  shape=torch.stack([scale[...,None]*(co[...,None]*a-si[...,None]*b),
                     scale[...,None]*(si[...,None]*a+co[...,None]*b)],-1).flatten(-2)/1000
  points=torch.cat([start[...,None,:],middle,end[...,None,:]],-2)
  path=(points[...,1:,:]-points[...,:-1,:]).square().sum(-1).sqrt().sum(-1)*scale/1000
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
