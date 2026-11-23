"""Extend the full-prediction audit to completed fresh historical trials."""
from pathlib import Path
import ast,hashlib
root=Path('E:/UTEC/Tesis/TesisIan/propuesta/replanteamiento_20261121')
p=root/'consolidar_auditoria_resultados.py'
s=p.read_text(encoding='utf-8-sig')
marker="for run,label in [('historica_fresca_s0','Mezcla histórica desde cero')"
if marker not in s:
 old="pending=[run for run in ['campo_multiescala_s0','aux_subviajes_s0','atencion_vial_s0'] if not (OUT/run/'resultado.json').exists()]"
 assert s.count(old)==1
 block="""for run,label in [('historica_fresca_s0','Mezcla histórica desde cero'),('control_historica_fresca_s0','Histórica: control sin memoria')]:
 source=OUT/run/'resultado.json'
 if source.exists():
  r=read(source);z=load_archive(source.parent/'validacion.npz')
  append(run,label+' — inicial nueva',z['epoch_0'],r['initial_validation'],source,kind='initial_candidate',learned=False)
  append(run,label+' — seleccionada',z['pred'],r['validation'],source,kind='selected_model',learned=r['best_epoch']>0)
  for row in r['all_epoch_validation']:
   epoch=row['epoch'];append(run,label+(' — época fija 20' if epoch==20 else ' — época '+str(epoch)),z['epoch_'+str(epoch)],row['validation'],source,kind='matched_fixed_epoch_20' if epoch==20 else 'epoch_checkpoint')
pending=[run for run in ['campo_multiescala_s0','aux_subviajes_s0','atencion_vial_s0','historica_fresca_s0','control_historica_fresca_s0'] if not (OUT/run/'resultado.json').exists()]"""
 updated=s.replace(old,block)
 ast.parse(updated)
 archive=root/'auditoria_antes_historica_113_outputs.py'
 assert not archive.exists()
 archive.write_bytes(p.read_bytes())
 p.write_text(updated,encoding='utf-8')
 print('Patched audit; previous source preserved; new SHA256',hashlib.sha256(p.read_bytes()).hexdigest())
else:print('Historical extension already present; no change.')
