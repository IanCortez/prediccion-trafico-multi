from base_replanteo import *
p=OUT/'verificar_conservacion.py';s=p.read_text();marker="# Verify source-complete nonparametric candidates under userthreshold."
if marker not in s:
 insert="""
# Verify source-complete nonparametric candidates under userthreshold.
file=OUT/'modelos_no_parametricos_protegidos.json'
if file.exists():
 report['nonparametric_qualified_candidates']=[]
 for candidate in read(file)['candidates']:
  run=OUT/candidate['id'];source=OUT/candidate['source_bank']
  assert source.resolve().is_relative_to(OUT.resolve()) and sha(source)==candidate['source_bank_sha256']
  assert sha(run/'validacion.npz')==candidate['predictions_sha256']
  for name,digest in candidate['sources'].items():assert sha(run/'fuentes'/name)==digest
  assert (OUT/candidate['required_parent']/'resultado.json').exists()
  report['nonparametric_qualified_candidates'].append(dict(id=candidate['id'],mae_s=candidate['validation']['mae_s'],source_complete=True,unchanged=True))
file=OUT/'modelos_atencion_subtrayectos_protegidos.json'
if file.exists():
 report['subtrajectory_branch_candidates']=[]
 for candidate in read(file)['candidates']:
  path=Path(candidate['model_file']).resolve()
  assert path.is_relative_to(OUT.resolve()) and sha(path)==candidate['model_sha256']
  for file,digest in candidate['dependencies'].items():assert sha(OUT/file)==digest
  report['subtrajectory_branch_candidates'].append(dict(id=candidate['id'],mae_s=candidate['validation']['mae_s'],unchanged=True))
"""
 assert s.count("write(OUT/'inventario_conservacion.json'")==1
 s=s.replace("write(OUT/'inventario_conservacion.json'",insert+"\nwrite(OUT/'inventario_conservacion.json'")
 p.write_text(s)
