"""Sourceelapsedtime is explicit; geometry and index context are arclength invariant."""
from base_replanteo import *
import torch
from tokens_subtrayectos_arco import SubtripArcLengthBank
class ArcTransferBank(SubtripArcLengthBank):
 @torch.no_grad()
 def features(self,xy_m,hour,weekend,parent,first,last,geometry_proxy=False):
  raw=super().features(xy_m,hour,weekend,parent,first,last).float()
  p=torch.as_tensor(parent,device=self.xy.device).long();i=torch.as_tensor(first,device=self.xy.device).long();j=torch.as_tensor(last,device=self.xy.device).long()
  valid=p>=0;p=p.clamp_min(0);offset=self.offset[p]
  begin=self.prefix_m[offset+i];finish=self.prefix_m[offset+j]
  full=self.prefix_m[offset+(self.n[p]-1).clamp_min(0)].clamp_min(1e-6)
  raw[...,28]=begin/full;raw[...,29]=(full-finish)/full;raw[...,33]=(finish-begin)/full
  if geometry_proxy:
   source=self.xy[offset+j]-self.xy[offset+i];source_chord=source.square().sum(-1).sqrt()
   query_chord=(xy_m[:,2:]-xy_m[:,:2]).square().sum(-1).sqrt()
   scale=(query_chord[:,None]/source_chord.clamp_min(1e-6)).clamp(.5,2)
   raw[...,16]=(source_chord/(25/3.6)*scale).clamp_min(.1).log()
   raw[...,32]=(full/(25/3.6)).clamp_min(.1).log()
  return torch.where(valid[...,None],raw,0.).to(torch.float16)
 @classmethod
 def from_arrays(cls,points,sizes,timestamps,device='cpu'):
  from datos_progreso_recurrente import lisbon_transitions
  obj=cls.__new__(cls);points=np.asarray(points,np.float32);sizes=np.asarray(sizes,np.int64)
  offsets=np.r_[0,np.cumsum(sizes)];assert offsets[-1]==len(points) and sizes.max()<=120
  prefix=np.zeros(len(points),np.float32)
  for a,b in zip(offsets[:-1],offsets[1:]):
   step=np.sqrt((np.diff(points[a:b].astype(float),axis=0)**2).sum(1))
   prefix[a:b]=np.r_[0,np.cumsum(step)]
  obj.xy=torch.tensor(points,device=device);obj.n=torch.tensor(sizes,device=device);obj.offset=torch.tensor(offsets,device=device)
  obj.ts=torch.tensor(timestamps,device=device);obj.prefix_m=torch.tensor(prefix,device=device)
  starts,ends=lisbon_transitions();obj.starts=torch.tensor(starts,device=device);obj.ends=torch.tensor(ends,device=device)
  obj.fractions=torch.arange(1,8,device=device,dtype=torch.float32)/8
  return obj
def compatibility_indices(names):
 shape=[[names.index('projected_'+kind+'_shape_'+str(i)) for i in range(14)] for kind in ['length','free']]
 available=[names.index('projected_'+kind+'_available') for kind in ['length','free']]
 return np.asarray(shape,np.int64),np.asarray(available,np.int64)
@torch.no_grad()
def add_public_compatibility(raw,x,shape_indices,available_indices):
 extra=[]
 for shape,available in zip(shape_indices,available_indices):
  distance=(raw[...,:14].float()-x[:,shape].float()[:,None,:]).square().mean(-1).sqrt()
  exists=x[:,available].float().clamp(0,1)
  extra.append(torch.where(exists[:,None]>.5,distance,torch.zeros_like(distance))[...,None])
 extra.extend([x[:,i].float().clamp(0,1)[:,None,None].expand(-1,64,-1) for i in available_indices])
 out=torch.cat([raw.float()]+extra,-1)
 return torch.where(raw[...,27:28]>.5,out,0.).to(torch.float16)
