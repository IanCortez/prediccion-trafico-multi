"""Canonical source-set order before neural operations; no targets/Validation consulted."""
from base_replanteo import *
import ast
probe=read(OUT/'diagnostico_orden_atencion.json')
assert probe['point_medians_exact'] and probe['max_absolute_probability_difference']<1e-6
path=OUT/'modelo_atencion_historia.py';s=path.read_text(encoding='utf-8-sig')
old="  R=raw.float();B,K,W=hidden.shape;valid=R[:,:,27]>.5;exists=valid.any(1)"
new="""  R=raw.float();B,K,W=hidden.shape
  # Full lexicographic canonicalization gives identical arithmetic for a source set.
  order=torch.arange(64,device=R.device)[None].expand(B,-1)
  for column in range(27,-1,-1):
   key=R[:,:,column].gather(1,order);permutation=torch.argsort(key,dim=1,stable=True)
   order=order.gather(1,permutation)
  R=R.gather(1,order[:,:,None].expand(-1,-1,28))
  valid=R[:,:,27]>.5;exists=valid.any(1)"""
assert s.count(old)==1;s=s.replace(old,new);ast.parse(s);path.write_text(s,encoding='utf-8')
write(OUT/'canonizacion_atencion_historia.json',dict(reason='Tiny floating-point tail changes under reordered sums; canonicalize before network instead of weakening the invariance gate.',
 before=probe,original_model_sha256=sha(OUT/'verificacion_atencion_pre_canon/modelo_atencion_historia.py'),
 updated_model_sha256=sha(path),query_or_source_values_changed=False,Validation_or_Test_access=False))
print('Source-set arithmetic canonicalized; original version and failed proof retained.')
