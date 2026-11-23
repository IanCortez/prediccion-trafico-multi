from base_replanteo import *
import ast
s=(OUT/'inferencia_atencion_subtrayectos.py').read_text()
s=s.replace('class FrozenSubtrajectoryExpert:','class FrozenArcPriorExpert:')
s=s.replace("run_id='atencion_subtrayectos_s0'","run_id='prior_arco_subtrayectos_s0'")
s=s.replace("module=frozen_module('modelo_atencion_subtrayectos');self.tokens=frozen_module('tokens_subtrayectos')",
 "module=frozen_module('modelo_prior_arco_subtrayectos');self.tokens=frozen_module('tokens_subtrayectos');arc_tokens=frozen_module('tokens_subtrayectos_arco')")
s=s.replace("self.model=module.SubtrajectoryAttention(","self.model=module.ArcPriorAttention(")
s=s.replace("self.model=self.model.cuda().eval();self.library=self.tokens.SubtripTokenBank()",
 "self.model=self.model.cuda().eval();self.library=self.tokens.SubtripTokenBank();self.arc_library=arc_tokens.SubtripArcLengthBank(shared=self.library)\n  assert read(OUT/'distancia_arco_puntos_Train_s0/resultado.json')['outputs_sha256']==r['config']['prefix_sha256']")
line="  raw=self.library.features(sq['xy_m'],sq['hour'],sq['weekend'],src[0],src[1],src[2])"
assert line in s;s=s.replace(line,line+"\n  geometry=self.arc_library.features(sq['xy_m'],sq['hour'],sq['weekend'],src[0],src[1],src[2])[:,:,:14]")
s=s.replace("old,raw[index])","old,raw[index],geometry[index])")
p=OUT/'inferencia_prior_arco_subtrayectos.py';assert not p.exists();ast.parse(s);p.write_text(s)
s=(OUT/'reproducir_atencion_subtrayectos.py').read_text()
s=s.replace('from inferencia_atencion_subtrayectos import FrozenSubtrajectoryExpert',
 'from inferencia_prior_arco_subtrayectos import FrozenArcPriorExpert')
s=s.replace('expert=FrozenSubtrajectoryExpert()','expert=FrozenArcPriorExpert()')
old="""  out=expert.model(*[torch.as_tensor(v[index],device='cuda') for v in [V,C,H]],old,sub)"""
new="""  geometry=expert.arc_library.features(*[torch.as_tensor(sq[k][index],device='cuda') for k in ['xy_m','hour','weekend']],
   S['parent'][index],S['first'][index],S['last'][index])[:,:,:14]
  out=expert.model(*[torch.as_tensor(v[index],device='cuda') for v in [V,C,H]],old,sub,geometry)"""
assert old in s;s=s.replace(old,new)
s=s.replace("sha(OUT/'inferencia_atencion_subtrayectos.py')","sha(OUT/'inferencia_prior_arco_subtrayectos.py')")
p=OUT/'reproducir_prior_arco_subtrayectos.py';assert not p.exists();ast.parse(s);p.write_text(s)
write(OUT/'inferencia_prior_arco_preparada.json',dict(prepared=True,not_run=True,
 sources={name:sha(OUT/name) for name in ['inferencia_prior_arco_subtrayectos.py','reproducir_prior_arco_subtrayectos.py']},
 scope='Qualifiedcheckpoint only; full127941quantiles plusraw65/reverse/singleton; no Test.'))
print('Independentcheckpointreload for thegeometryprior-onlycorrection ready.')
