"""Freeze the next serial experiment and reconcile current operational bookkeeping."""
from base_replanteo import *
import ast
proof=read(OUT/'verificacion_mediana_directa_cpu.json')
assert proof['passed'] and proof['source_sha256']==sha(OUT/'modelo_mediana_directa.py')
assert read(OUT/'piloto_reloj_fino_train_s0/resultado.json')['passed'] is False
p=OUT/'concluir_historica_fresca.py';s=p.read_text(encoding='utf-8-sig')
old="""saved=[]
for item in read(OUT/'candidatos_vivos_historica.json').get('models',[]):
 saved.append(item)
# The authoritative run manifest also proves every retained treatment checkpoint.
for item in ra['saved_checkpoints']:
 path=Path(item.get('model_file',item.get('file','')))
 if not path.is_absolute():path=OUT/ids[0]/path
 if path.is_file() and item.get('sha256'):
  assert sha(path)==item['sha256']"""
new="""protected=[]
for name,r in results.items():
 for item in r['saved_checkpoints']:
  path=(OUT/name/item['file']).resolve();assert path.is_relative_to((OUT/name).resolve())
  qualifies=item['validation']['mae_s']<131.06
  if qualifies or path.is_file():
   assert path.is_file() and sha(path)==item['sha256']
   if qualifies:protected.append(dict(experiment=name,epoch=item['epoch'],file=str(path),sha256=item['sha256'],validation=item['validation']))
write(OUT/'modelos_historicos_completos_protegidos.json',dict(candidates=protected,preserve=True,all_completed_metrics_predictions_retained=True,Test='sealed'))"""
if old in s:s=s.replace(old,new);ast.parse(s);p.write_text(s,encoding='utf-8')
assert 'modelos_historicos_completos_protegidos.json' in s
guard_names=['modelo_mediana_directa.py','entrenar_mediana_directa.py','inferencia_historica_fresca.py','certificar_historica_fresca.py','concluir_historica_fresca.py','plan_mediana_directa.json','base_replanteo.py','verificacion_mediana_directa_cpu.json']
for name in guard_names:
 if name.endswith('.py'):ast.parse((OUT/name).read_text(encoding='utf-8-sig'))
write(OUT/'secuencia_mediana_fuentes_congeladas.json',{name:sha(OUT/name) for name in guard_names})
state=read(OUT/'CONTINUIDAD_OPERATIVA.json')
state['previous_goal_turn_classification']='VERIFIED WAIT: inspected live controlPID24952/session10237 and protected artifacts. Current turn: audit expanded154outputs; Train-only fine-clock gate negative; direct MAE architecture numerically verified and serial next trial frozen.'
state['queued_next_trial']=dict(id='mediana_directa_s0',source_guard='secuencia_mediana_fuentes_congeladas.json',GPU='Only after current historical control terminal, certified and compared.',runner='secuencia_mediana.ps1',numerical_proof_passed=True,query='unchanged five fields',Test='sealed')
state['completed_fine_clock_gate']=dict(passed=False,held_Train_n=50000,delta_mae_s=-.08457910733315234,ci95_s=[-.2970998252529686,.11660568243583629],action='Do not train full primary model using this cycle bank.')
state['full_prediction_audit']=dict(outputs=154,archives=24,reported_metric_error=0,rows_labels_timestamps_exact=True)
if 'provisional_candidate' in state:state['provisional_candidate']['status']='COMPLETED and fully certified; best27 MAE130.204821. No longer provisional or running.'
state['resource_policy']='One heavy GPU at a time; next immutable-reference point cache <10MiB, moments computed in256rowchunks. Current project15.44GiB; original data and protected models intact.'
state['next_after_result']=['Monitor current control24952/session10237 and serial runner process. Do not restart because an observation times out.','Serial runner waits for actual control termination, requires complete result, certifies control, compares fixed20 and selected epochs, verifies every qualifying historical checkpoint, then launches frozen mediana_directa_s0.','No queued/active source edits; inspect logs and exact process state. All completed metrics/predictions preserved.','After direct median completes: audit full predictions, verify and certify any learned best, run matched frozen-student control if genuine improvement, draw conclusion and continue toward73.5. Test closed.']
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
print(json.dumps(dict(frozen_sources=len(guard_names),next='serial certification, comparison, direct-median experiment',Test='sealed')))
