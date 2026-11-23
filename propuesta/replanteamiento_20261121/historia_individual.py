"""Deterministic five-field query -> individual historical-source token features."""
from base_replanteo import *
import torch,pandas as pd
from scipy.spatial import cKDTree
from historicos_rutas import query_xy
TOKEN_NAMES=['history_shape_'+str(i) for i in range(14)]+['scaled_history_path_km','history_log_circuity','log_scaled_history_time_s',
 'source_minus_query_o_x','source_minus_query_o_y','source_minus_query_d_x','source_minus_query_d_y','history_OD4D_km',
 'clock_difference_sin','clock_difference_cos','weekend_match','log_chord_scale','log_retrieval_prior','history_available']
assert len(TOKEN_NAMES)==28
def query_parts(d):
 local=pd.to_datetime(d['timestamp'],unit='s',utc=True).tz_convert('Europe/Lisbon')
 return dict(xy_km=query_xy(d).astype(np.float32),hour=(np.asarray(local.hour)+np.asarray(local.minute)/60+np.asarray(local.second)/3600).astype(np.float32),
  weekend=np.asarray(local.dayofweek)>=5)
class IndividualHistoryBank:
 def __init__(self,path,expected_sha256):
  assert sha(path)==expected_sha256;self.memory=dict(np.load(path));self.trees={};self.devices={}
 def query_indices(self,d,training=False):
  m=self.memory;xy=query_xy(d);fold=(d['timestamp']//604800)%5 if training else np.full(len(xy),5)
  result=np.full((len(xy),64),-1,np.int32)
  if not training:assert (m['timestamp']+m['time_s']).max()<int(np.min(d['timestamp']))
  for f in np.unique(fold):
   f=int(f);take=np.flatnonzero(fold==f)
   if f not in self.trees:
    source=np.flatnonzero(m['fold']!=f) if f<5 else np.arange(len(m['row']))
    self.trees[f]=(cKDTree(m['xy'][source]),source)
   tree,source=self.trees[f];distance,index=tree.query(xy[take],k=64,workers=2);near=source[index].astype(np.int32);near[distance>.5]=-1
   valid=near>=0
   if f<5:assert (m['fold'][near[valid]]!=f).all()
   result[take]=near
  return result
 def tensors(self,device):
  key=str(torch.device(device))
  if key not in self.devices:
   self.devices[key]={k:torch.as_tensor(self.memory[k],device=device,dtype=torch.float32) for k in ['xy','geometry','length_m','time_s','hour','weekend']}
  return self.devices[key]
 @torch.no_grad()
 def features(self,xy,hour,weekend,indices):
  device=xy.device;m=self.tensors(device);ids=indices.long();available=ids>=0;ids=ids.clamp_min(0)
  source=m['xy'][ids];delta=source-xy[:,None,:]
  qd=xy[:,2:]-xy[:,:2];sd=source[...,2:]-source[...,:2]
  qnorm=qd.square().sum(-1).clamp_min(1e-12).sqrt();snorm=sd.square().sum(-1).clamp_min(1e-12).sqrt()
  scale=qnorm[:,None]/snorm;co=(qd[:,None]*sd).sum(-1)/(qnorm[:,None]*snorm)
  si=(qd[:,None,1]*sd[...,0]-qd[:,None,0]*sd[...,1])/(qnorm[:,None]*snorm)
  g=m['geometry'][ids].reshape(len(xy),64,7,2);a=g[...,0];b=g[...,1]
  shape=torch.stack([scale[...,None]*(co[...,None]*a-si[...,None]*b),scale[...,None]*(si[...,None]*a+co[...,None]*b)],-1).flatten(-2)
  path=m['length_m'][ids]*scale/1000;circuity=(path/qnorm[:,None]).clamp_min(.01).log()
  time=(m['time_s'][ids]*scale).clamp_min(.1).log()
  distance=delta.square().sum(-1).sqrt()
  clock=torch.remainder(m['hour'][ids]-hour[:,None]+12,24)-12;phase=clock*(2*np.pi/24)
  match=(m['weekend'][ids]==weekend[:,None].float()).float()
  prior=-.5*(distance/.25).square()-.5*(clock/4).square()+torch.where(match>.5,0.,float(np.log(.35)))
  raw=torch.cat([shape,path[...,None],circuity[...,None],time[...,None],delta,distance[...,None],
   phase.sin()[...,None],phase.cos()[...,None],match[...,None],scale.clamp_min(1e-9).log()[...,None],prior[...,None],available.float()[...,None]],-1)
  assert raw.shape==(len(xy),64,28)
  return torch.where(available[...,None],raw,0.).to(torch.float16)
 def raw(self,items,device='cpu',training=False):
  from consulta_od import od_records
  d=od_records(items);part=query_parts(d);ids=self.query_indices(d,training)
  tensor={k:torch.as_tensor(v,device=device) for k,v in part.items()}
  return self.features(tensor['xy_km'],tensor['hour'],tensor['weekend'],torch.as_tensor(ids,device=device))
@torch.no_grad()
def compress_source_values(raw,history):
 """Same64tokens, but historical G/path/time replaced by existing6prototype summaries."""
 r=raw.float();h=history.float();valid=r[:,:,27]>.5;available=h[:,:,23]>.5
 distance=(r[:,:,None,:14]-h[:,None,:,:14]).reshape(len(r),64,6,7,2).square().sum(-1).sqrt().mean(-1)
 distance=distance.masked_fill(~available[:,None],torch.inf);group=distance.argmin(-1)
 prototype=h.gather(1,group[...,None].expand(-1,-1,26));out=r.clone()
 out[:,:,:16]=prototype[:,:,:16];out[:,:,16]=(prototype[:,:,16]*600).clamp_min(.1).log()
 valid=valid&available.any(-1)[:,None];out=torch.where(valid[...,None],out,0.);out[:,:,27]=valid
 return out.to(raw.dtype)
