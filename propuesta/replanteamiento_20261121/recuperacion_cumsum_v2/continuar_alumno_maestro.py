"""Launch the predeclared soft-target student after actual preceding process exits."""
from base_replanteo import *
import ctypes,subprocess
from ctypes import wintypes
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
k.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.WaitForSingleObject.restype=wintypes.DWORD
k.CloseHandle.argtypes=[wintypes.HANDLE]
k.QueryFullProcessImageNameW.argtypes=[wintypes.HANDLE,wintypes.DWORD,wintypes.LPWSTR,ctypes.POINTER(wintypes.DWORD)]
handles=[]
for pid in [10844,8372]:
 h=k.OpenProcess(0x00100000|0x1000,False,pid)
 if h:
  buf=ctypes.create_unicode_buffer(32768);size=wintypes.DWORD(len(buf));assert k.QueryFullProcessImageNameW(h,0,buf,ctypes.byref(size))
  assert Path(buf.value).resolve()==Path(sys.executable).resolve(),'Unexpected process identity; do not guess/restart'
  handles.append((pid,h))
plan=read(OUT/'plan_alumno_maestro.json')
for f,s in plan['sources'].items():assert sha(OUT/f)==s
assert sha(OUT/'entrenar_aux_solicitud.py')==plan['control_source_sha256']
write(OUT/'pipeline_alumno_estado.json',dict(stage='Waiting for original teacher CPU and request-study supervisor handles; no additional GPU',original_pids=[p for p,h in handles],supervisor_pid=os.getpid(),goal='active',Test='sealed'))
for pid,h in handles:
 while k.WaitForSingleObject(h,0)==258:time.sleep(15)
 k.CloseHandle(h)
required=[OUT/'maestro_cruzado_s0/resultado.json',OUT/'aux_solicitud_control_s0/resultado.json',OUT/'aux_solicitud_semantic_s0/resultado.json']
if not all(f.exists() for f in required):
 write(OUT/'pipeline_alumno_estado.json',dict(stage='Preceding stage incomplete or a risk candidate takes priority; inspect existing results before launching another GPU.',missing=[str(f.relative_to(ROOT)) for f in required if not f.exists()],goal='active',Test='sealed'));exit()
teacher,control,semantic=[read(f) for f in required]
assert teacher['all_eligible_rows_assigned_once'] and teacher['own_whole_fold_excluded_for_every_row']
if not .72<=teacher['full_out_of_fit_Train_distribution_scores']['coverage80']<=.90:
 write(OUT/'pipeline_alumno_estado.json',dict(stage='Full teacher calibration outside declared range; no student GPU',goal='active',Test='sealed'));exit()
if min(control['validation']['mae_s'],semantic['validation']['mae_s'])<73.5:
 write(OUT/'pipeline_alumno_estado.json',dict(stage='Existing Validation target candidate needs freeze/reproduction/confirmation; no new training.',goal='active',Test='sealed'));exit()
for item in teacher['folds']:assert sha(OUT/'maestro_cruzado_s0'/('maestro_fold_'+str(item['fold'])+'.cbm'))==item['model_sha256']
assert read(OUT/'verificacion_perdida_maestro.json')['passed']
for f,s in plan['sources'].items():assert sha(OUT/f)==s,'Do not modify queued sources'
run=OUT/'alumno_maestro_s0'
if not (run/'resultado.json').exists():
 args=[sys.executable,'-B',str(OUT/'entrenar_alumno_maestro.py'),'--arm','distill']
 if run.exists():
  assert (run/'ultima_reanudacion.pt').exists(),'Inspect failed initial gate before resuming';args+=['--resume']
 with (OUT/'alumno_maestro_s0.log').open('a' if run.exists() else 'w',encoding='utf-8') as log:
  child=subprocess.Popen(args,cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
  active=dict(experiment=run.name,pid=child.pid,supervisor_pid=os.getpid(),device='GPU',GPU_jobs=1)
  c=read(OUT/'CONTINUIDAD_OPERATIVA.json');c['current_trial']=active;write(OUT/'CONTINUIDAD_OPERATIVA.json',c)
  write(OUT/'pipeline_alumno_estado.json',dict(stage='Soft-target student GPU active, mandatory initial gate',active=active,goal='active',Test='sealed'))
  while child.poll() is None:time.sleep(15)
 if not (run/'resultado.json').exists():
  write(OUT/'pipeline_alumno_estado.json',dict(stage='Student ended without complete result; inspect actual log/checkpoint, no automatic overwrite',exit_code=child.returncode,goal='active',Test='sealed'));exit()
r=read(run/'resultado.json');student=dict(np.load(run/'validacion.npz'));base=dict(np.load(OUT/'aux_solicitud_control_s0/validacion.npz'))
for name in ['row','timestamp','true']:assert np.array_equal(student[name],base[name])
kept=[]
for item in r['saved_checkpoints']:
 assert sha(run/item['file'])==item['sha256'];kept.append(dict(id=run.name,**item,reproduction_pending=True))
write(OUT/'candidatos_conservados_alumno_maestro.json',dict(checkpoints=kept,original_reference_protected=True,Test='sealed'))
write(OUT/'conclusion_alumno_maestro.json',dict(student_result=r,matched_control=control['id'],control_result_file=str(required[1].relative_to(ROOT)),final_fixed_epoch_student_vs_control=paired(base['true'],student['epoch_4'],base['epoch_4'],base['timestamp']),best_selected_student_vs_control=paired(base['true'],student['pred'],base['pred'],base['timestamp']),retained_checkpoints=kept,limitations='Privileged teacher was Train-only. Student uses original OD inputs and true labels. Reused Validation selection is exploratory.',goal='active',Test='sealed',next='Reproduce promising models and continue distinct improvements if confirmation target is unmet. Individual study completion does not complete the goal.'))
write(OUT/'pipeline_alumno_estado.json',dict(stage='Student study complete: critically review full results and continue goal',goal='active',Test='sealed'))
print(json.dumps(dict(stage='soft_target_student_complete',goal='active',Test='sealed')),flush=True)
