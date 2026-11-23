from base_replanteo import *
import ast
p=OUT/'verificar_conservacion.py';s=p.read_text();marker='# Full joint source-model candidates, each with complete dependency verification.'
if marker not in s:
 insert="""
# Full joint source-model candidates, each with complete dependency verification.
file=OUT/'modelos_subtrayectos_conjuntos_protegidos.json'
if file.exists():
 report['joint_subtrajectory_candidates']=[]
 for candidate in read(file)['candidates']:
  path=Path(candidate['model_file']).resolve()
  assert path.is_relative_to(OUT.resolve()) and sha(path)==candidate['model_sha256']
  for name,digest in candidate['dependencies'].items():assert sha(OUT/name)==digest
  result=read(OUT/candidate['id']/'resultado.json')
  assert sha(OUT/candidate['id']/'validacion.npz')==result['predictions_sha256']
  report['joint_subtrajectory_candidates'].append(dict(id=candidate['id'],mae_s=candidate['validation']['mae_s'],unchanged=True))
"""
 s=s.replace("write(OUT/'inventario_conservacion.json'",insert+"\nwrite(OUT/'inventario_conservacion.json'")
 ast.parse(s);p.write_text(s)
p=OUT/'consolidar_auditoria_resultados.py';s=p.read_text();marker='# Matched joint source-value control, if actually completed.'
if marker not in s:
 insert="""
# Matched joint source-value control, if actually completed.
source=OUT/'control_subtrayectos_conjuntos_s0/resultado.json'
if source.exists():
 r=read(source);z=load_archive(source.parent/'validacion.npz')
 append(source.parent.name,'Subtrayectos conjuntos: control sin valores',z['pred'],r['validation'],source)
"""
 s=s.replace('pending=[',insert+'\npending=[');ast.parse(s);p.write_text(s)
for name in ['concluir_subtrayectos_conjuntos.py','inferencia_subtrayectos_conjuntos.py','reproducir_subtrayectos_conjuntos.py','entrenar_control_subtrayectos_conjuntos.py']:ast.parse((OUT/name).read_text(),filename=name)
write(OUT/'instrumentos_conjuntos_preparados.json',dict(syntax_valid=True,
 frozen_before_treatment_validation=True,sources={name:sha(OUT/name) for name in ['concluir_subtrayectos_conjuntos.py','inferencia_subtrayectos_conjuntos.py','reproducir_subtrayectos_conjuntos.py','entrenar_control_subtrayectos_conjuntos.py']},
 status='Preparedonly; runafteractualterminalresult orpreservationtrigger. Neverreruntraining on observationtimeout.',Test='sealed'))
print('Closedresultaudit, preservation and independentinference tools ready.')
