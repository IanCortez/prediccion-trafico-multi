"""Compatibility-only recovery after verified pre-update failure; no active jobs."""
from base_replanteo import *
import shutil
failed=OUT/'aux_solicitud_control_s0'
assert not list(failed.glob('*.pt')) and not (failed/'historia.json').exists() and not (failed/'validacion.npz').exists()
assert sha(BEST/'modelo.pt')=='55fa2262e3b26b259839fa06e6f28812e75ff30b578aceb6022c8286c2cb161f'
archive=OUT/'recuperacion_cumsum_v2';assert not archive.exists();archive.mkdir()
files=['entrenar_aux_solicitud.py','preparar_alumno_maestro.py','entrenar_alumno_maestro.py','continuar_aux_solicitud.py','continuar_alumno_maestro.py','plan_aux_solicitud.json','plan_alumno_maestro.json','pipeline_aux_solicitud_estado.json','pipeline_alumno_estado.json']
for name in files:shutil.copy2(OUT/name,archive/name)
shutil.copy2(OUT/'aux_solicitud_control_s0.log',archive/'fallo_previo.log')
changed={}
p=OUT/'entrenar_aux_solicitud.py';s=p.read_text(encoding='utf-8-sig')
assert s.count('torch.use_deterministic_algorithms(True)')==1
s=s.replace('torch.use_deterministic_algorithms(True)','torch.use_deterministic_algorithms(False)')
assert s.count("+'_s0')")==1;s=s.replace("+'_s0')","+'_s0_v2')")
s=s.replace("\nsources=[","\nconfig.update(training_determinism='Fixed seeds and standard CUDA algorithms, matching original reference; no bitwise training guarantee',compatibility_fix='Strict CUDA cumsum unsupported; failure occurred before any optimizer update')\nsources=[",1)
changed[p]=s
p=OUT/'continuar_aux_solicitud.py';s=p.read_text(encoding='utf-8-sig')
s=s.replace('for pid in [34284,29428]:','for pid in []:')
s=s.replace("+'_s0')","+'_s0_v2')")
s=s.replace('aux_solicitud_control_s0','aux_solicitud_control_s0_v2').replace('aux_solicitud_semantic_s0','aux_solicitud_semantic_s0_v2')
changed[p]=s
p=OUT/'preparar_alumno_maestro.py';s=p.read_text(encoding='utf-8-sig')
s=s.replace("+'_s0')","+'_s0_v2')")
s=s.replace('aux_solicitud_control_s0','aux_solicitud_control_s0_v2').replace('alumno_maestro_s0','alumno_maestro_s0_v2')
s=s.replace("'primary_train_rule']:","'primary_train_rule','training_determinism','compatibility_fix']:")
changed[p]=s
p=OUT/'continuar_alumno_maestro.py';s=p.read_text(encoding='utf-8-sig')
s=s.replace('import ctypes,subprocess','import ctypes,subprocess,argparse\nap=argparse.ArgumentParser();ap.add_argument("--wait-request-pid",type=int,required=True);wait_args=ap.parse_args()')
s=s.replace('for pid in [10844,8372]:','for pid in [wait_args.wait_request_pid]:')
s=s.replace('aux_solicitud_control_s0','aux_solicitud_control_s0_v2').replace('aux_solicitud_semantic_s0','aux_solicitud_semantic_s0_v2').replace('alumno_maestro_s0','alumno_maestro_s0_v2')
changed[p]=s
for p,s in changed.items():compile(s,str(p),'exec')
for p,s in changed.items():p.write_text(s,encoding='utf-8')
plan=read(OUT/'plan_aux_solicitud.json')
plan['sources']['entrenar_aux_solicitud.py']=sha(OUT/'entrenar_aux_solicitud.py')
plan['compatibility_generation']='v2, fresh arm directories; all previous jobs confirmed terminal'
plan['numerics_recovery']='Standard seeded CUDA FP32 as reference. Strict deterministic CUDA cumsum is unsupported in this installed build; no bitwise training guarantee. Primary likelihood, RPS, geometry and class losses unchanged.'
plan['failed_initial_stage']='aux_solicitud_control_s0: no optimizer update, no checkpoint or predictions. Exact source/log archived.'
write(OUT/'plan_aux_solicitud.json',plan)
write(failed/'fallo_tecnico.json',dict(stage='Initial GPU objective, before backward/optimizer updates',weights_updated=False,no_checkpoints_or_predictions=True,cause='Unsupported strict deterministic CUDA cumsum',replacement='aux_solicitud_control_s0_v2',original_source_sha256=sha(failed/'fuentes/entrenar_aux_solicitud.py')))
write(archive/'recuperacion.json',dict(all_previous_processes_confirmed_terminal=True,failed_trial_had_no_updates=True,reference_unchanged=True,model_losses_and_data_unchanged=True,changed_sources={p.name:sha(p) for p in changed},GPU_gate_mandatory=True,Test='sealed',goal='active'))
print('Compatibilidad corregida sin cambiar datos ni pérdidas; fuentes fallidas conservadas')
