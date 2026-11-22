"""Sequential post-training conclusions, one GPU calibration and one CPU-only pilot."""
from base_replanteo import *
import subprocess,ctypes,shutil
from ctypes import wintypes
kernel=ctypes.WinDLL('kernel32',use_last_error=True)
kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
kernel.OpenProcess.restype=wintypes.HANDLE
kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
kernel.CloseHandle.argtypes=[wintypes.HANDLE]
def alive(pid):
 h=kernel.OpenProcess(0x1000,False,pid)
 if not h:return False
 value=wintypes.DWORD();ok=kernel.GetExitCodeProcess(h,ctypes.byref(value));kernel.CloseHandle(h)
 return bool(ok and value.value==259)
parent=OUT/'espacial_fases_referencia_s0'
while alive(35416):
 write(OUT/'pipeline_estado.json',dict(stage='Waiting for verified live training process',pid=35416,supervisor_pid=os.getpid(),goal='active',Test='sealed',last_update_unix=time.time()))
 time.sleep(15)
if not (parent/'resultado.json').exists():
 write(OUT/'pipeline_estado.json',dict(stage='Training process ended without complete result; inspect log and resume checkpoint',goal='active',requires_user_input=False))
 print('Parent ended without result. Research goal remains active; inspect and resume.',flush=True)
 exit()
r=read(parent/'resultado.json')
assert r['validation']['n']==127941 and r['reference_validation']['n']==127941
assert sha(parent/'modelo.pt')==r['model_sha256']
write(OUT/'conclusion_fases_antes_calibracion.json',dict(validation=r['validation'],reference=r['reference_validation'],graph=r['graph_validation'],paired=r['paired_vs_retained'],preserve_threshold_s=131.06,qualifies=r['qualifies_for_preservation'],new_primary_best=r['new_primary_best'],Test='sealed',conclusion='Full Validation result of trained model, recorded before any weight calibration. Goal not complete without frozen-candidate confirmation.'))
print(json.dumps(dict(event='parent_complete',validation=r['validation'],reference=r['reference_validation'],goal='active')),flush=True)
if min(r['validation']['mae_s'],r['graph_validation']['mae_s'])<73.5:
 write(OUT/'pipeline_estado.json',dict(stage='Eligible validation candidate requires reproduction/freeze/confirmation audit; no new research job launched',goal='active',Test='sealed'))
 exit()
children=[];logs=[]
for name,script,device in [('calibracion_unica_fases_s0','calibrar_fases_una_vez.py','GPU'),('piloto_entorno_compartido_s0','piloto_entorno_compartido.py','CPU')]:
 if (OUT/name/'resultado.json').exists():continue
 assert not (OUT/name).exists(),'Existing incomplete stage must be inspected, not overwritten'
 log=(OUT/(name+'.log')).open('w',encoding='utf-8');logs.append(log)
 process=subprocess.Popen([sys.executable,'-B',str(OUT/script)],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
 child=dict(experiment=name,pid=process.pid,script=script,device=device,process=process)
 children.append(child);print(json.dumps(dict(event='started',experiment=name,pid=process.pid,device=device)),flush=True)
while children:
 active=[]
 for c in children[:]:
  code=c['process'].poll()
  if code is None:active.append({k:c[k] for k in ['experiment','pid','script','device']});continue
  result=OUT/c['experiment']/'resultado.json'
  print(json.dumps(dict(event='finished',experiment=c['experiment'],exit_code=code,complete_result=result.exists())),flush=True)
  if result.exists():
   sources=OUT/c['experiment']/'fuentes';sources.mkdir(exist_ok=True)
   for filename in [c['script'],'base_replanteo.py']+(['verosimilitud_entorno.py'] if c['device']=='CPU' else []):shutil.copy2(OUT/filename,sources/filename)
  children.remove(c)
 write(OUT/'pipeline_estado.json',dict(stage='Post-training evidence in progress' if children else 'Post-training stages ended; inspect results and continue goal',active=active,goal='active',max_GPU_jobs=1,Test='sealed',last_update_unix=time.time()))
 if children:time.sleep(15)
for log in logs:log.close()
print('Post-training stages finished; goal remains ACTIVE. Inspect candidates and controlled evidence; continue until confirmed target.',flush=True)

