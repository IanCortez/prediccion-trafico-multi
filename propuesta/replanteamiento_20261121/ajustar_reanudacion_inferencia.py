from base_replanteo import *
import ast
p=OUT/'evaluar_prefijo_acelerado.py'
assert not (OUT/'prefijo_nativo_s0/evaluacion_acelerada').exists()
s=p.read_text(encoding='utf-8-sig')
a="write(audit/'seleccion_lote.json',"
b="if (audit/'progreso_predicciones.npz').exists():\n    selected_batch=read(audit/'protocolo.json')['config']['canonical_query_batch']\n    assert any(r['batch']==selected_batch and r['eligible'] for r in bench)\nwrite(audit/'seleccion_lote.json',"
assert a in s;s=s.replace(a,b)
ast.parse(s);p.write_text(s,encoding='utf-8')
