"""Raw-five-field inference for a qualified, source-frozen, full joint model."""
from base_replanteo import *
import torch,importlib,argparse
class FrozenJointSubtrajectory:
 def __init__(self,run_id='subtrayectos_conjuntos_s0'):
  run=(OUT/run_id).resolve();assert run.parent==OUT.resolve()
  r=read(run/'resultado.json');assert r['complete'] and r['qualified']
  self.run=run;self.record=r
  for name,digest in r['sources'].items():assert sha(run/'fuentes'/name)==digest
  path=Path(r['model_file']).resolve();assert path.is_relative_to(run) and sha(path)==r['model_sha256']
  sys.path.insert(0,str(run/'fuentes'))
  modules={}
  for name in ['modelo_mezcla_proyeccion','modelo_mezcla_historica','modelo_subtrayectos_conjuntos',
               'tokens_subtrayectos','recuperar_subtrayectos','historicos_rutas','inferencia_historica_fresca']:
   m=importlib.import_module(name);assert Path(m.__file__).resolve()==(run/'fuentes'/(name+'.py')).resolve();modules[name]=m
  from modelo_mezcla_rutas import make_duration_model
  ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['epoch']==24 and ck['config']==r['config']
  self.base_config=ck['base_config'];self.features=ck['features'];ctor=ck['constructor']
  saved=dict(np.load(run/'constructor_tokens.npz'))
  for key in saved:assert np.array_equal(saved[key],ctor[key])
  duration=make_duration_model(ck['base_constructor'],ck['base_config']);mc=ck['map_constructor'];hc=ck['history_constructor']
  projection=modules['modelo_mezcla_proyeccion'].ProjectionMixture(duration,mc['mean'],mc['std'],'expanded')
  parent=modules['modelo_mezcla_historica'].HistoricalRouteMixture(projection,hc['mean'],hc['std'],hc['time_anchor'],'historical')
  self.model=modules['modelo_subtrayectos_conjuntos'].JointSubtrajectoryAttention(parent,ctor['mean'],ctor['std'],ctor['time_anchor'],'individual')
  self.model.load_state_dict(ck['state'],strict=True);self.model=self.model.cuda().eval()
  self.tokens=modules['tokens_subtrayectos'];self.library=self.tokens.SubtripTokenBank()
  self.memory=modules['recuperar_subtrayectos'].SubtrajectoryMemory();self.memory.build_tree(5)
  self.history=modules['historicos_rutas'].HistoryRouteBank(OUT/'datos_prototipos_historicos_s0/memoria_Train.npz')
  self.raw_inputs=modules['inferencia_historica_fresca'].raw_inputs
  self.control=r['config'].get('mode')=='matched_source_value_control'
 def adjust_tokens(self,raw):
  if self.control:
   raw=raw.clone();raw[...,:17]=self.model.token_mean[:17].to(raw.dtype);raw[...,32]=self.model.token_mean[32].to(raw.dtype)
  return raw
 @torch.no_grad()
 def raw(self,items):
  assert isinstance(items,list) and len(items)>0
  X,C,H,names=self.raw_inputs(items,self.base_config,self.history);assert names==self.features
  dd={key:np.array([item[key] for item in items]) for key in ['o_lon','o_lat','d_lon','d_lat','timestamp']}
  src=self.memory.query(dd);part=self.tokens.query_parts_sub(dd)
  sq={key:torch.as_tensor(value,device='cuda') for key,value in part.items()}
  raw=self.adjust_tokens(self.library.features(sq['xy_m'],sq['hour'],sq['weekend'],src[0],src[1],src[2]))
  parts=[]
  for first in range(0,len(items),2048):
   n=min(2048,len(items)-first);index=torch.arange(first,first+n,device='cuda')
   if n<2048:index=torch.cat([index,index[-1:].expand(2048-n)])
   arrays=[torch.as_tensor(value,device='cuda')[index] for value in [X,C,H]]
   out=self.model(*arrays,raw[index]);parts.append(self.model.duration.quantiles(out[0])[:n].cpu().numpy())
  return np.concatenate(parts)
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',default='subtrayectos_conjuntos_s0');p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 access=deny_test();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction(.5)
 source=Path(a.input).resolve();target=Path(a.output).resolve()
 assert source.is_relative_to(ROOT.resolve()) and target.is_relative_to(ROOT.resolve()) and not target.exists()
 expert=FrozenJointSubtrajectory(a.run);q=expert.raw(read(source))
 write(target,dict(model_id=a.run,model_sha256=expert.record['model_sha256'],input_fields=['o_lon','o_lat','d_lon','d_lat','timestamp'],
  predictions=[dict(input_index=i,q10_s=float(v[0]),median_s=float(v[1]),q90_s=float(v[2])) for i,v in enumerate(q)],
  uncertainty='Exploratory marginal intervals; not independent calibration.',Test_access=False))
if __name__=='__main__':main()
