"""Reproducible five-field inference of a qualified frozen-parent subtrajectory expert."""
from base_replanteo import *
import torch,importlib.util
from inferencia_atencion_historia import load_frozen_attention,raw_inputs
class FrozenSubtrajectoryExpert:
 def __init__(self,run_id='atencion_subtrayectos_s0',allow_pending=False):
  run=(OUT/run_id).resolve();assert run.parent==OUT.resolve()
  self.run=run
  if allow_pending:
   assert not (run/'resultado.json').exists()
   config=read(run/'protocolo.json');history=read(run/'historia.json')
   assert len(history)==6 and all(v['primary_train_n']==1023485 for v in history)
   path=run/'epoca_6.pt';checkpoint=torch.load(path,map_location='cpu',weights_only=False)
   recovery=torch.load(run/'ultima_reanudacion.pt',map_location='cpu',weights_only=False)
   assert checkpoint['epoch']==recovery['epoch']==6 and checkpoint['config']==config==recovery['config']
   for key in checkpoint['branch_state']:assert torch.equal(checkpoint['branch_state'][key],recovery['branch_state'][key])
   sources={p.name:sha(p) for p in (run/'fuentes').glob('*.py')}
   for name,digest in sources.items():assert sha(OUT/name)==digest
   r=dict(complete=False,qualified=False,config=config,model_file=str(path),model_sha256=sha(path),sources=sources)
   write(run/'recuperacion_evaluacion_protocolo.json',dict(reload_epoch=6,model_sha256=r['model_sha256'],saved_branch_matches_final_optimizer_checkpoint=True,
    training_not_repeated=True,reason='FinalB2048distribution calculation exceeded6GiBcap withallTrain tensors resident; reloadonlyValidationdata.',
    same_architecture_weights_batch_and_query_information=True,Test='sealed'))
  else:
   r=read(run/'resultado.json');assert r['complete'] and r['qualified']
  self.record=r
  for name,digest in r['sources'].items():assert sha(run/'fuentes'/name)==digest
  p=Path(r['model_file']).resolve();assert p.is_relative_to(run) and sha(p)==r['model_sha256']
  parent,ck,ref,_,proto,oldlibrary,helper,oldtoken=load_frozen_attention(r['config']['parent_id'],device='cpu')
  assert ref['model_sha256']==r['config']['parent_sha256']
  self.base_config=ck['base_config'];self.features=ck['features'];self.proto=proto;self.oldlibrary=oldlibrary;self.helper=helper;self.oldtoken=oldtoken
  sys.path.insert(0,str(run/'fuentes'))
  def frozen_module(name):
   spec=importlib.util.spec_from_file_location('qualified_'+name,run/'fuentes'/(name+'.py'))
   module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module
  module=frozen_module('modelo_atencion_subtrayectos');self.tokens=frozen_module('tokens_subtrayectos')
  source=importlib.import_module('recuperar_subtrayectos')
  assert Path(source.__file__).resolve()==(run/'fuentes/recuperar_subtrayectos.py').resolve()
  state=torch.load(p,map_location='cpu',weights_only=False);constructor=state['constructor']
  assert state['config']==r['config'] and state['epoch']==6 and state['features']==self.features
  saved=dict(np.load(run/'constructor_tokens.npz'))
  for key in saved:assert np.array_equal(constructor[key],saved[key])
  self.model=module.SubtrajectoryAttention(parent,constructor['mean'],constructor['std'],constructor['time_anchor'],'individual')
  check=self.model.load_state_dict(state['branch_state'],strict=False)
  assert not check.unexpected_keys and all(k.startswith('parent.') for k in check.missing_keys)
  self.model=self.model.cuda().eval();self.library=self.tokens.SubtripTokenBank()
  self.memory=source.SubtrajectoryMemory();self.memory.build_tree(5)
 @torch.no_grad()
 def raw(self,items):
  X,C,H,Q,names=raw_inputs(items,self.base_config,self.proto,self.oldlibrary,self.helper,self.oldtoken)
  assert names==self.features
  dd={k:np.array([v[k] for v in items]) for k in ['o_lon','o_lat','d_lon','d_lat','timestamp']}
  src=self.memory.query(dd);sq={k:torch.as_tensor(v,device='cuda') for k,v in self.tokens.query_parts_sub(dd).items()}
  raw=self.library.features(sq['xy_m'],sq['hour'],sq['weekend'],src[0],src[1],src[2])
  parts=[]
  for first in range(0,len(items),2048):
   n=min(2048,len(items)-first);index=torch.arange(first,first+n,device='cuda')
   if n<2048:index=torch.cat([index,index[-1:].expand(2048-n)])
   oldq={k:torch.as_tensor(v,device='cuda')[index] for k,v in Q.items()}
   old=self.oldlibrary.features(oldq['xy_km'],oldq['hour'],oldq['weekend'],oldq['indices'])
   out=self.model(*[torch.as_tensor(v,device='cuda')[index] for v in [X,C,H]],old,raw[index])
   parts.append(self.model.duration.quantiles(out[0])[:n].cpu().numpy())
  return np.concatenate(parts)
