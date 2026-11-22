"""Conclude the full native trial, then run a conditional CPU gate before another GPU job."""
from base_replanteo import *
import ctypes,subprocess
from ctypes import wintypes
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
k.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.WaitForSingleObject.restype=wintypes.DWORD
k.CloseHandle.argtypes=[wintypes.HANDLE]
handles=[]
for pid in [20612,33120]:
    h=k.OpenProcess(0x00100000|0x1000,False,pid)
    if h:handles.append(h)
while any(k.WaitForSingleObject(h,0)==258 for h in handles):
    write(OUT/'pipeline_historia_estado.json',dict(stage='Waiting for frozen native full result and actual GPU exit',original_processes=[20612,33120],supervisor_pid=os.getpid(),goal='active',Test='sealed'))
    time.sleep(15)
for h in handles:k.CloseHandle(h)
parent=OUT/'prefijo_nativo_s0'
if not (parent/'resultado.json').exists():
    write(OUT/'pipeline_historia_estado.json',dict(stage='Native process ended without complete result: inspect log, resume or inference-only recovery; do not train a successor yet',goal='active',Test='sealed'))
    print('Native result incomplete; inspect actual failure. Goal ACTIVE.',flush=True)
    exit()
r=read(parent/'resultado.json')
assert r['validation']['n']==127941 and all(m['n']==127941 for m in r['all_alpha_validation'])
assert sha(parent/'modelo.pt')==r['model_sha256']
keep=[]
for alpha,m in zip(r['config']['alphas'],r['all_alpha_validation']):
    if alpha>0 and m['mae_s']<131.25671207822356:
        keep.append(dict(id='prefijo_nativo_s0',alpha=alpha,validation=m,checkpoint=str(parent/'modelo.pt'),model_sha256=r['model_sha256'],reference_sha256=r['reference_sha256'],full_prediction_archive=str(parent/'validacion.npz'),threshold_qualifies=m['mae_s']<131.06,reproduction_and_512particle_robustness_pending=True))
write(OUT/'candidatos_conservados_prefijo_nativo.json',dict(candidates=keep,original_reference_always_protected=True,TransferTraj_unchanged=True,Test='sealed'))
write(OUT/'conclusion_prefijo_nativo.json',dict(validation=r['validation'],graph_validation=r['graph_validation'],all_outputs=r['all_alpha_validation'],selected_alpha=r['selected_alpha'],paired=r['paired_vs_retained'],best_epoch=r['best_epoch'],goal='active',Test='sealed',conclusion='Complete native-generator evidence before deciding another trial; alpha0 is the protected baseline, not a learned gain. Additional temporal history requires a matched conditional pilot.'))
print(json.dumps(dict(stage='native_complete',validation=r['validation'],graph=r['graph_validation'],selected_alpha=r['selected_alpha'],candidates=keep)),flush=True)
if min(m['mae_s'] for m in r['all_alpha_validation'])<73.5:
    write(OUT/'pipeline_historia_estado.json',dict(stage='Eligible Validation candidate: reproduce, freeze and confirm; do not launch further research',goal='active',Test='sealed'))
    exit()
stages=[('piloto_historia_condicional_s0','piloto_historia_condicional.py','CPU')]
for name,script,device in stages:
    assert not (OUT/name).exists(),'Inspect existing incomplete stage instead of overwriting it'
    log=(OUT/(name+'.log')).open('w',encoding='utf-8')
    p=subprocess.Popen([sys.executable,'-B',str(OUT/script)],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    print(json.dumps(dict(stage='conditional_CPU_pilot_started',pid=p.pid)),flush=True)
    while p.poll() is None:
        write(OUT/'pipeline_historia_estado.json',dict(stage='Conditional history CPU pilot',pid=p.pid,supervisor_pid=os.getpid(),device=device,goal='active',Test='sealed'))
        time.sleep(15)
    log.close()
    if p.returncode!=0 or not (OUT/name/'resultado.json').exists():
        write(OUT/'pipeline_historia_estado.json',dict(stage='Conditional CPU pilot failed technically; inspect log before continuing',exit_code=p.returncode,goal='active',Test='sealed'))
        exit()
pilot=read(OUT/'piloto_historia_condicional_s0/resultado.json')
print(json.dumps(dict(stage='conditional_pilot_complete',results=pilot['results'],history_minus_native=pilot['history_minus_native'],history_minus_control=pilot['history_minus_matched_control'],worth_GPU=pilot['worth_GPU_test'])),flush=True)
if not pilot['worth_GPU_test']:
    write(OUT/'pipeline_historia_estado.json',dict(stage='Conditional pilot did not justify temporal-prior GPU training; continue with a genuinely different mechanism, do not repeat blend or smoothing searches',goal='active',Test='sealed',requires_user_input=False))
    exit()
run=OUT/'prefijo_historia_s0'
assert not run.exists()
log=(OUT/'prefijo_historia_s0.log').open('w',encoding='utf-8')
p=subprocess.Popen([sys.executable,'-B',str(OUT/'entrenar_prefijo_historia.py'),'--id','prefijo_historia_s0'],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
active=dict(experiment='prefijo_historia_s0',pid=p.pid,device='GPU',supervisor_pid=os.getpid(),max_GPU_jobs=1)
for name in ['sesiones_activas.json','sesiones_extra.json']:write(OUT/name,dict(active=[active],goal='active',Test='sealed'))
c=read(OUT/'CONTINUIDAD_OPERATIVA.json');c['current_trial']=active;c['conditional_pilot']=pilot;write(OUT/'CONTINUIDAD_OPERATIVA.json',c)
print(json.dumps(dict(stage='history_GPU_started',**active)),flush=True)
while p.poll() is None:
    write(OUT/'pipeline_historia_estado.json',dict(stage='Historical prefix model in progress',active=active,goal='active',Test='sealed'))
    time.sleep(15)
log.close()
write(OUT/'pipeline_historia_estado.json',dict(stage='Read complete history result and continue goal' if (run/'resultado.json').exists() else 'Inspect history training failure before resuming',exit_code=p.returncode,goal='active',Test='sealed'))
print('History stage ended; goal remains ACTIVE.',flush=True)
