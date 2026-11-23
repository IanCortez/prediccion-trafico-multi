from base_replanteo import *
import ast
old=OUT/'tokens_subtrayectos.py';source=old.read_text()
method=source[source.index(' @torch.no_grad()\n def features'):]
oldblock="""  pos=first[...,None]+(last-first)[...,None]*self.fractions
  lo=pos.long();hi=torch.minimum(lo+1,last[...,None]);f=pos-lo
  middle=self.xy[offset[...,None]+lo]*(1-f[...,None])+self.xy[offset[...,None]+hi]*f[...,None]"""
newblock="""  begin=self.prefix_m[offset+first];finish=self.prefix_m[offset+last]
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
  middle=self.xy[offset[...,None]+lo]*(1-f[...,None])+self.xy[offset[...,None]+hi]*f[...,None]"""
assert oldblock in method;method=method.replace(oldblock,newblock)
oldpath="""  points=torch.cat([start[...,None,:],middle,end[...,None,:]],-2)
  path=(points[...,1:,:]-points[...,:-1,:]).square().sum(-1).sqrt().sum(-1)*scale/1000"""
assert oldpath in method;method=method.replace(oldpath,"  path=total_distance*scale/1000")
header="""\"""Historical34Dtokens withgeometry consistent withoriginalnormalizedarclengthteacher.\"""
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
"""
s=header+method;ast.parse(s)
p=OUT/'tokens_subtrayectos_arco.py';assert not p.exists();p.write_text(s)
write(OUT/'plan_correccion_arco_subtrayectos.json',dict(
 reason='Verified512Train semanticmismatch: oldmedianpoint error212.32m andmedianpathlength understatement10.75percent vsoriginalteacher.',
 correction='Sample sourcepolyline bymetricarclength; use fullsourceintervallength, notninepointapproximation. 34Dlayout unchanged.',
 invariant='Originaltimes15*(j-i), sourceindices, sourceclock, endpointdelta, availability, allcolumns16..33, raw5query andprimarycohort unchanged.',
 old_model='129.990034 remains valid foritsfrozen time-sampledmethod; neverreplace itsinputfunction.',
 future_comparison='Separate6fixedepochbranch vsoriginaltime-sampled6epochbranch, samefrozen30componentparent andsameinitialneuralparameters. No epoch/neighbor/bandwidth sweep.',
 normalizer='Same32768Trainqueryindices; recompute first16geometry statistics only, preserve allotherstatistics andtimeanchor exactly.',
 resource='OneGPUmaximum; prepareCPUsemanticproof first. Sharexy sourceTensor, prefixadditional<=163MiB uncompressed; inferencepreflightbeforeTrain tensorsresident.',
 active_trial='Current24epoch joint-time-sampled model continuesunchanged; correction sources separate, not launched.',
 Test='SEALED'))
print('Separatearclength-source implementation prepared; existingmodel/inputfiles unchanged.')
