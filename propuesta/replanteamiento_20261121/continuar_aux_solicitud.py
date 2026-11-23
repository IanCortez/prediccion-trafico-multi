"""Run the fixed matched auxiliary study only after the actual risk chain exits."""
from base_replanteo import *
import ctypes,subprocess
from ctypes import wintypes
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
k.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.WaitForSingleObject.restype=wintypes.DWORD
k.CloseHandle.argtypes=[wintypes.HANDLE]
k.QueryFullProcessImageNameW.argtypes=[wintypes.HANDLE,wintypes.DWORD,wintypes.LPWSTR,ctypes.POINTER(wintypes.DWORD)]
handles=[]
for pid in []:
 h=k.OpenProcess(0x00100000|0x1000,False,pid)
 if h:
  buf=ctypes.create_unicode_buffer(32768);size=wintypes.DWORD(len(buf));assert k.QueryFullProcessImageNameW(h,0,buf,ctypes.byref(size))
  assert Path(buf.value).resolve()==Path(sys.executable).resolve(),'Unexpected process identity; inspect before launch'
  handles.append((pid,h))
plan=read(OUT/'plan_aux_solicitud.json')
for f,s in plan['sources'].items():assert sha(OUT/f)==s
write(OUT/'pipeline_aux_solicitud_estado.json',dict(stage='Waiting on original risk process handles; no second GPU',waiting_original_pids=[p for p,h in handles],supervisor_pid=os.getpid(),goal='active',Test='sealed'))
for pid,h in handles:
 while k.WaitForSingleObject(h,0)==258:time.sleep(15)
 k.CloseHandle(h)
risk=OUT/'riesgo_total_s0/resultado.json'
if not risk.exists():
 write(OUT/'pipeline_aux_solicitud_estado.json',dict(stage='Risk ended without complete result. Inspect/recover it before new GPU.',goal='active',Test='sealed'));exit()
rr=read(risk);assert all(m['n']==127941 for m in rr['all_alpha_validation'])
if any(m['mae_s']<131.25671207822356 for alpha,m in zip(rr['config']['alphas'],rr['all_alpha_validation']) if alpha>0):
 write(OUT/'pipeline_aux_solicitud_estado.json',dict(stage='Improved risk candidate takes priority: reproduce/freeze it before auxiliary GPU study.',goal='active',Test='sealed'));exit()
assert read(OUT/'verificacion_aux_solicitud_cpu.json')['passed']
records=[];kept=[]
for arm in ['control','semantic']:
 run=OUT/('aux_solicitud_'+arm+'_s0_v2')
 if not (run/'resultado.json').exists():
  for f,s in plan['sources'].items():assert sha(OUT/f)==s,'Do not modify queued sources'
  args=[sys.executable,'-B',str(OUT/'entrenar_aux_solicitud.py'),'--arm',arm]
  if run.exists():
   assert (run/'ultima_reanudacion.pt').exists(),'Inspect incomplete gate before recovery';args+=['--resume']
  with (OUT/(run.name+'.log')).open('a' if run.exists() else 'w',encoding='utf-8') as log:
   child=subprocess.Popen(args,cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
   active=dict(experiment=run.name,pid=child.pid,supervisor_pid=os.getpid(),device='GPU',GPU_jobs=1)
   c=read(OUT/'CONTINUIDAD_OPERATIVA.json');c['current_trial']=active;write(OUT/'CONTINUIDAD_OPERATIVA.json',c)
   write(OUT/'pipeline_aux_solicitud_estado.json',dict(stage='Active fixed auxiliary arm',active=active,goal='active',Test='sealed'))
   while child.poll() is None:time.sleep(15)
  if not (run/'resultado.json').exists():
   write(OUT/'pipeline_aux_solicitud_estado.json',dict(stage='Arm ended without complete result; inspect log/checkpoint, no automatic overwrite',exit_code=child.returncode,arm=arm,goal='active',Test='sealed'));exit()
 r=read(run/'resultado.json');records.append(r)
 for item in r['saved_checkpoints']:
  assert sha(run/item['file'])==item['sha256']
  kept.append(dict(id=run.name,**item,reproduction_pending=True))
 write(OUT/'candidatos_conservados_aux_solicitud.json',dict(checkpoints=kept,original_reference_protected=True,Test='sealed'))
 if r['validation']['mae_s']<73.5:
  write(OUT/'pipeline_aux_solicitud_estado.json',dict(stage='Target Validation candidate: freeze and reproduce before confirmation; no more training',goal='active',Test='sealed'));exit()
control=dict(np.load(OUT/'aux_solicitud_control_s0_v2/validacion.npz'));semantic=dict(np.load(OUT/'aux_solicitud_semantic_s0_v2/validacion.npz'))
assert np.array_equal(control['row'],semantic['row'])
write(OUT/'conclusion_aux_solicitud.json',dict(results=records,final_fixed_epoch_semantic_vs_control=paired(control['true'],semantic['epoch_4'],control['epoch_4'],control['timestamp']),best_selected_semantic_vs_control=paired(control['true'],semantic['pred'],control['pred'],control['timestamp']),retained_checkpoints=kept,goal='active',Test='sealed',next='Critically review full results, reproduce every promising checkpoint and continue distinct research if target is unmet. Never close goal because this study finished.'))
write(OUT/'pipeline_aux_solicitud_estado.json',dict(stage='Matched study complete: inspect conclusions and continue goal',goal='active',Test='sealed'))
print(json.dumps(dict(stage='matched_auxiliary_study_complete',goal='active',Test='sealed')),flush=True)
