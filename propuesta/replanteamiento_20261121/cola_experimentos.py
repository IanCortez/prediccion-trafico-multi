"""Local bounded-concurrency experiment queue; never closes the research goal."""
from base_replanteo import *
import subprocess,ctypes
from ctypes import wintypes
kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE;kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)];kernel.CloseHandle.argtypes=[wintypes.HANDLE]
def alive(pid):
 h=kernel.OpenProcess(0x1000,False,pid)
 if not h:return False
 value=wintypes.DWORD();ok=kernel.GetExitCodeProcess(h,ctypes.byref(value));kernel.CloseHandle(h);return bool(ok and value.value==259)
active=[dict(experiment='descompuesto_aux0_s0',pid=38580,session_id=61148),dict(experiment='espacial_semimarkov_s0_v2',pid=32092,session_id=13808)]
queue=[dict(experiment='memoria_residual_oof_s0',script='entrenar_residual.py',args=['--id','memoria_residual_oof_s0']),dict(experiment='espacial_direccional_s0',script='entrenar_direccional.py',args=['--id','espacial_direccional_s0'])]
children={};completed=[];handles=[]
while active or queue:
 for r in active[:]:
  child=children.get(r['pid']);live=child.poll() is None if child else alive(r['pid'])
  if not live:
   result=OUT/r['experiment']/'resultado.json';r=dict(r,completed_result=result.exists(),exit_code=child.returncode if child else None);completed.append(r);active=[z for z in active if z['pid']!=r['pid']];print(json.dumps(dict(event='finished',**r)),flush=True)
 while queue and len(active)<2:
  r=queue.pop(0);assert not (OUT/r['experiment']).exists();log=(OUT/(r['experiment']+'.log')).open('w',encoding='utf-8');handles.append(log);p=subprocess.Popen([sys.executable,'-B',str(OUT/r['script']),*r['args']],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT);children[p.pid]=p;r=dict(r,pid=p.pid);active.append(r);print(json.dumps(dict(event='started',**r)),flush=True)
 write(OUT/'sesiones_activas.json',dict(status='active research goal',supervisor_pid=os.getpid(),active=active,pending=queue,completed_in_queue=completed,max_simultaneous_GPU_jobs=2,Test='sealed',overall_goal_must_continue=True,last_update_unix=time.time()))
 if active or queue:time.sleep(15)
for f in handles:f.close()
print('Queue finished; research goal remains ACTIVE; inspect conclusions and plan next alternatives.',flush=True)

