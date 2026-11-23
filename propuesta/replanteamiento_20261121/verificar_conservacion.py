from base_replanteo import *
start=time.time();access=deny_test();prev=read(BASE/'desarrollo_20261121/verificacion_conservados.json');assert sha(BEST/'modelo.pt')==prev['retained_best_sha256'];checks=[]
for r in prev['TransferTraj_prediction_archives_unchanged']:
 candidates=[p for p in (BASE/'comparaciones_transfertraj').rglob('*.npz') if p.stem==r['id'] or p.parent.name==r['id']];assert len(candidates)==1,(r['id'],candidates);got=sha(candidates[0]);assert got==r['sha256'];checks.append(dict(id=r['id'],sha256=got,unchanged=True))
total=0;count=0;new_total=0;stack=[ROOT];skipped=[]
while stack:
 directory=stack.pop()
 for entry in os.scandir(directory):
  p=Path(entry.path)
  if not p.resolve().is_relative_to(ROOT):skipped.append(str(p));continue
  if entry.is_dir(follow_symlinks=False):stack.append(p)
  elif entry.is_file(follow_symlinks=False):
   size=entry.stat(follow_symlinks=False).st_size;total+=size;count+=1
   if p.is_relative_to(OUT):new_total+=size
report=dict(project_bytes=total,project_GiB=total/1024**3,file_count=count,new_work_bytes=new_total,new_work_MiB=new_total/1024**2,note='Logical file size snapshot while training is active; checkpoints and logs can change during measurement',best_model_unchanged=True,best_model_sha256=prev['retained_best_sha256'],TransferTraj_predictions_unchanged=checks,original_data_were_not_modified=True,Test_files_opened=False,paths_skipped_outside_root=skipped,seconds=time.time()-start)
current=read(OUT/'nuevo_mejor_verificado.json');current_path=Path(current['model_file']).resolve()
assert current_path.is_relative_to(ROOT.resolve()) and sha(current_path)==current['model_sha256']
report.update(current_best_unchanged=True,current_best_model_sha256=current['model_sha256'],current_best_mae_s=current['validation']['mae_s'],
 legacy_best_fields_refer_to_protected_original_reference=True)
lineagefile=OUT/'modelos_mejores_protegidos.json'
if lineagefile.exists():
 report['verified_best_lineage_unchanged']=[]
 for model in read(lineagefile):
  path=Path(model['model_file']).resolve()
  assert path.is_relative_to(OUT.resolve()) and sha(path)==model['model_sha256']
  report['verified_best_lineage_unchanged'].append(dict(id=model['id'],sha256=model['model_sha256'],unchanged=True))
livefile=OUT/'candidatos_vivos_historica.json'
if livefile.exists():
 live=read(livefile);report['live_qualified_candidates']=[]
 for candidate in live['candidates']:
  path=Path(candidate['model_file']).resolve()
  assert path.is_relative_to(OUT.resolve()) and sha(path)==candidate['model_sha256']
  report['live_qualified_candidates'].append(dict(epoch=candidate['epoch'],mae_s=candidate['validation']['mae_s'],sha256=candidate['model_sha256'],unchanged=True))
report['qualified_control_models']=[]
for manifest in ['candidatos_vivos_control_historica.json','modelos_historicos_completos_protegidos.json']:
 path=OUT/manifest
 if path.exists():
  for candidate in read(path)['candidates']:
   file=Path(candidate.get('model_file',candidate.get('file'))).resolve()
   expected=candidate.get('model_sha256',candidate.get('sha256'))
   assert file.is_relative_to(OUT.resolve()) and sha(file)==expected
   report['qualified_control_models'].append(dict(manifest=manifest,epoch=candidate['epoch'],sha256=expected,unchanged=True))
report['completed_direct_models']=[]
for name in ['mediana_directa_s0','control_mediana_directa_s0']:
 source=OUT/name/'resultado.json'
 if source.exists():
  r=read(source);assert sha(source.parent/'validacion.npz')==r['predictions_sha256']
  for candidate in r['saved_checkpoints']:
   if candidate['validation']['mae_s']>=131.06:continue
   file=(source.parent/candidate['file']).resolve();assert file.is_relative_to(source.parent.resolve()) and sha(file)==candidate['sha256']
   report['completed_direct_models'].append(dict(id=name,epoch=candidate['epoch'],sha256=candidate['sha256'],unchanged=True))
report['individual_history_branch_checkpoints']=[]
for name in ['atencion_historia_individual_s0','control_compresion_historia_s0']:
 manifest=OUT/name/'candidatos_conservados.json'
 if manifest.exists():
  for candidate in read(manifest)['candidates']:
   file=(manifest.parent/candidate['file']).resolve()
   assert file.is_relative_to(manifest.parent.resolve()) and sha(file)==candidate['sha256']
   report['individual_history_branch_checkpoints'].append(dict(id=name,epoch=candidate['epoch'],mae_s=candidate['validation']['mae_s'],sha256=candidate['sha256'],unchanged=True))

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


# Qualified geometryprior-only branches and their source-complete dependencies.
file=OUT/'modelos_prior_arco_protegidos.json'
if file.exists():
 report['arclength_prior_candidates']=[]
 for candidate in read(file)['candidates']:
  path=Path(candidate['model_file']).resolve();assert path.is_relative_to(OUT.resolve()) and sha(path)==candidate['model_sha256']
  for name,digest in candidate['dependencies'].items():assert sha(OUT/name)==digest
  report['arclength_prior_candidates'].append(dict(id=candidate['id'],mae_s=candidate['validation']['mae_s'],unchanged=True))

write(OUT/'inventario_conservacion.json',report);print(json.dumps({k:report[k] for k in ['project_GiB','file_count','new_work_MiB','best_model_unchanged','current_best_unchanged','current_best_mae_s','seconds']}))

