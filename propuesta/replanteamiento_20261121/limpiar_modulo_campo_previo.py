from base_replanteo import *
p=OUT/'modelo_campo_multiescala.py';s=p.read_text(encoding='utf-8-sig')
old="return (self.embedding(ids)*self.confidence[ids,:,None] if False else self.embedding(ids)*self.confidence[ids][...,None]).mul(w[...,None]).sum(1)"
new="return (self.embedding(ids)*self.confidence[ids][...,None]*w[...,None]).sum(1)"
assert old in s;s=s.replace(old,new);compile(s,str(p),'exec');p.write_text(s,encoding='utf-8')
print('Rama muerta retirada; operación bilineal sin cambios')
