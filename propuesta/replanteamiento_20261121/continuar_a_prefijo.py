"""Wait on original Windows process HANDLES, then record evidence and launch one GPU trial."""
from base_replanteo import *
import ctypes,subprocess
from ctypes import wintypes
kernel=ctypes.WinDLL('kernel32',use_last_error=True)
kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
kernel.OpenProcess.restype=wintypes.HANDLE
kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
kernel.WaitForSingleObject.restype=wintypes.DWORD
kernel.CloseHandle.argtypes=[wintypes.HANDLE]
handles=[]
for pid in [2056,30780]:
    h=kernel.OpenProcess(0x00100000|0x1000,False,pid)
    if h:handles.append((pid,h))
# Holding handles means a later, unrelated reuse of a numeric PID cannot block us.
while any(kernel.WaitForSingleObject(h,0)==258 for _,h in handles):
    write(OUT/'pipeline_prefijo_estado.json',dict(stage='Waiting for actual preceding GPU and supervisor exit',original_pids=[p for p,h in handles],supervisor_pid=os.getpid(),goal='active',Test='sealed'))
    time.sleep(10)
for pid,h in handles:kernel.CloseHandle(h)
cal=OUT/'calibracion_unica_fases_s0'
assert (cal/'resultado.json').exists(),'Previous stage incomplete; inspect its log, never overwrite'
r=read(cal/'resultado.json')
assert all(m['n']==127941 for m in r['full_validation'])
threshold=131.06
retained=[]
parent=OUT/'espacial_fases_referencia_s0'
assert sha(parent/'modelo.pt')==r['shared_checkpoint_sha256']
for alpha,m in zip(r['alphas'],r['full_validation']):
    if alpha>0 and m['mae_s']<131.25671207822356:
        retained.append(dict(id='calibracion_unica_fases_s0',alpha=alpha,validation=m,checkpoint=str(parent/'modelo.pt'),checkpoint_sha256=r['shared_checkpoint_sha256'],reference=str(BEST/'modelo.pt'),reference_sha256=r['reference_sha256'],qualifies_threshold=m['mae_s']<threshold,predictions=str(cal/'predicciones_distintas.npz'),reproduction_pending=True))
write(OUT/'candidatos_conservados_tras_calibracion.json',dict(candidates=retained,protected_original_model=read(BEST/'resultado.json')['model_sha256'],preserve_rule='Keep every complete result below131.06s and every new best; all new bests need reproduction audit',Test='sealed'))
summary=dict(alphas=r['alphas'],full_validation=r['full_validation'],selected_by_fixed_subset=r['selected_by_fixed_subset'],selected_full_validation=r['selected_full_validation'],qualified=r['qualified_candidates'],best_full_exploratory_alpha=r['best_full_validation_exploratory_alpha'],conclusion='This is the ONLY predeclared calibration of this fixed physical branch. Do not keep changing weights. Alpha0 reproduces the protected model and is not a learned improvement.',goal='active',Test='sealed')
write(OUT/'conclusion_calibracion_unica.json',summary)
print(json.dumps(dict(stage='calibration_complete',selected_alpha=r['selected_by_fixed_subset'],selected_validation=r['selected_full_validation'],qualified=r['qualified_candidates'],retained=retained)),flush=True)
if min(m['mae_s'] for m in r['full_validation'])<73.5:
    write(OUT/'pipeline_prefijo_estado.json',dict(stage='Threshold candidate requires reproduction freeze and sealed confirmation; no new training',goal='active',Test='sealed'))
    exit()
pilot=read(OUT/'piloto_entorno_compartido_s0/resultado.json')
assert not pilot['worth_GPU_shared_effect_test']
assert read(OUT/'verificacion_prefijo_nativo.json')['passed']
run=OUT/'prefijo_nativo_s0'
assert not run.exists(),'Do not overwrite or duplicate an existing trial'
log=(OUT/'prefijo_nativo_s0.log').open('w',encoding='utf-8')
p=subprocess.Popen([sys.executable,'-B',str(OUT/'entrenar_prefijo_nativo.py'),'--id','prefijo_nativo_s0'],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
active=dict(experiment='prefijo_nativo_s0',pid=p.pid,device='GPU',supervisor_pid=os.getpid(),max_GPU_jobs=1,resume_command='Local Conda python -B entrenar_prefijo_nativo.py --id prefijo_nativo_s0 --resume; only after verified process exit')
for name in ['sesiones_activas.json','sesiones_extra.json']:write(OUT/name,dict(active=[active],goal='active',Test='sealed'))
continuity=read(OUT/'CONTINUIDAD_OPERATIVA.json')
continuity['current_trial']=active
continuity['calibration_conclusion']=summary
continuity['model_protection_file']='candidatos_conservados_tras_calibracion.json'
write(OUT/'CONTINUIDAD_OPERATIVA.json',continuity)
print(json.dumps(dict(event='native_prefix_training_started',**active)),flush=True)
while p.poll() is None:
    write(OUT/'pipeline_prefijo_estado.json',dict(stage='Native prefix model in progress',active=active,goal='active',Test='sealed',last_update_unix=time.time()))
    time.sleep(15)
log.close()
complete=(run/'resultado.json').exists()
write(OUT/'pipeline_prefijo_estado.json',dict(stage='Read native prefix result, preserve qualifiers and continue research' if complete else 'Inspect failed native prefix log; resume when appropriate, never overwrite',exit_code=p.returncode,complete_result=complete,goal='active',Test='sealed'))
print(json.dumps(dict(event='native_prefix_stage_ended',exit_code=p.returncode,complete_result=complete,goal='active')),flush=True)
