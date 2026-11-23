from base_replanteo import *
p=OUT/'rutas_proyeccion_incierta.py';code=p.read_text(encoding='utf-8-sig')
old="cost=edge_distance(oe[q,a],of[q,a],de[q,b],df[q,b],u,v,length if kind==0 else free,DL if kind==0 else DF,*(CL if kind==0 else CF))"
new="""if kind==0:cost=edge_distance(oe[q,a],of[q,a],de[q,b],df[q,b],u,v,length,DL,*CL)
     else:cost=edge_distance(oe[q,a],of[q,a],de[q,b],df[q,b],u,v,free,DF,*CF)"""
assert old in code;p.write_text(code.replace(old,new),encoding='utf-8')
p=OUT/'piloto_incertidumbre_proyeccion.py';code=p.read_text(encoding='utf-8-sig')
assert "run=OUT/'piloto_proyeccion_incierta_s0'" in code
p.write_text(code.replace("run=OUT/'piloto_proyeccion_incierta_s0'","run=OUT/'piloto_proyeccion_incierta_s0_v2'"),encoding='utf-8')
write(OUT/'piloto_proyeccion_incierta_s0/fallo_tecnico.json',dict(stage='Numba compilation before any scientific output',error='Conditional tuple expansion unsupported bytecode',recovery='Explicit length/free branches, no math/data/threshold change',replacement='piloto_proyeccion_incierta_s0_v2',Test_access=False))
print('Compiler-only compatibility fix; no model update or scientific output from failed attempt.')
