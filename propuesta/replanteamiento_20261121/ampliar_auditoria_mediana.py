"""Prepare future result audit/protection; does not edit any active or queued sources."""
from pathlib import Path
import ast
root=Path('E:/UTEC/Tesis/TesisIan/propuesta/replanteamiento_20261121')
p=root/'consolidar_auditoria_resultados.py';s=p.read_text(encoding='utf-8-sig')
marker="for run,label in [('mediana_directa_s0','Mediana explícita con ajuste conjunto')"
if marker not in s:
 needle="pending=[run for run in ['campo_multiescala_s0','aux_subviajes_s0','atencion_vial_s0','historica_fresca_s0','control_historica_fresca_s0'] if not (OUT/run/'resultado.json').exists()]"
 assert s.count(needle)==1
 block="""for run,label in [('mediana_directa_s0','Mediana explícita con ajuste conjunto'),('control_mediana_directa_s0','Mediana: representación congelada')]:
 source=OUT/run/'resultado.json'
 if source.exists():
  r=read(source);z=load_archive(source.parent/'validacion.npz')
  append(run,label+' — inicial',z['epoch_0'],r['initial_validation'],source,kind='initial_candidate',learned=False)
  append(run,label+' — seleccionada',z['pred'],r['validation'],source,kind='selected_model',learned=r['learned_update_selected'])
  for row in r['all_epoch_validation']:
   epoch=row['epoch'];append(run,label+(' — final fija' if epoch==6 else ' — época '+str(epoch)),z['epoch_'+str(epoch)],row['validation'],source,kind='fixed_final_epoch' if epoch==6 else 'epoch_checkpoint')
pending=[run for run in ['campo_multiescala_s0','aux_subviajes_s0','atencion_vial_s0','historica_fresca_s0','control_historica_fresca_s0','mediana_directa_s0'] if not (OUT/run/'resultado.json').exists()]"""
 s=s.replace(needle,block);ast.parse(s);p.write_text(s,encoding='utf-8')
p=root/'verificar_conservacion.py';s=p.read_text(encoding='utf-8-sig')
if "qualified_control_models" not in s:
 needle="write(OUT/'inventario_conservacion.json',report);"
 block="""report['qualified_control_models']=[]
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
write(OUT/'inventario_conservacion.json',report);"""
 assert s.count(needle)==1;s=s.replace(needle,block);ast.parse(s);p.write_text(s,encoding='utf-8')
print('Historical control and future direct-median audits/protection prepared; frozen experiment sources untouched.')
