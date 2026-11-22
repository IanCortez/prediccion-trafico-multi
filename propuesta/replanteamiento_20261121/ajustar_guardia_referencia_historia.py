from base_replanteo import *
import ast
p=OUT/'entrenar_prefijo_historia.py'
assert not (OUT/'prefijo_historia_s0').exists()
s=p.read_text(encoding='utf-8-sig')
a="a=alpha*support\n                p=(1-a[:,None])*reference[ii]+a[:,None]*path\n                p=p/p.sum(-1,keepdim=True)"
b="if alpha==0:\n                    p=reference[ii]\n                else:\n                    a=alpha*support\n                    p=(1-a[:,None])*reference[ii]+a[:,None]*path\n                    p=p/p.sum(-1,keepdim=True)"
assert a in s
s=s.replace(a,b)
ast.parse(s)
p.write_text(s,encoding='utf-8')
r=read(OUT/'preparacion_entrenador_historia.json')
r['target_sha256']=sha(p)
r['exact_alpha0_reference_branch']=True
write(OUT/'preparacion_entrenador_historia.json',r)
print('Nueva variante conserva alfa0 exactamente; entrenamiento actual sin cambios.')
