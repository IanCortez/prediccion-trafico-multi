"""Bound inference moment temporaries before the new trial; no active source edited."""
from pathlib import Path
import ast
p=Path('E:/UTEC/Tesis/TesisIan/propuesta/replanteamiento_20261121/modelo_mediana_directa.py')
s=p.read_text(encoding='utf-8-sig')
old="""  p=out[4].exp();grid=self.grid/600
  component_mean=(p*grid).sum(-1)
  component_sd=(p*(grid-component_mean[...,None]).square()).sum(-1).clamp_min(1e-8).sqrt()
  context=torch.cat([hidden[0],out[1].exp(),component_mean,component_sd],-1)"""
new="""  moments=[];grid=self.grid/600
  for first in range(0,len(x),256):
   p=out[4][first:first+256].exp()
   mean=(p*grid).sum(-1)
   sd=(p*(grid-mean[...,None]).square()).sum(-1).clamp_min(1e-8).sqrt()
   moments.append(torch.cat([out[1][first:first+256].exp(),mean,sd],-1))
  context=torch.cat([hidden[0],torch.cat(moments,0)],-1)"""
assert s.count(old)==1
s=s.replace(old,new);ast.parse(s);p.write_text(s,encoding='utf-8')
print('Moment temporaries bounded to256 rows; numerical proof must be refreshed.')
