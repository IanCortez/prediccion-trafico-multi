"""Conclude preceding model/control stages, then test a different full-trip objective."""
from base_replanteo import *
import ctypes,subprocess
from ctypes import wintypes
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
k.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.WaitForSingleObject.restype=wintypes.DWORD
k.CloseHandle.argtypes=[wintypes.HANDLE]
k.QueryFullProcessImageNameW.argtypes=[wintypes.HANDLE,wintypes.DWORD,wintypes.LPWSTR,ctypes.POINTER(wintypes.DWORD)]
h=k.OpenProcess(0x00100000|0x1000,False,34908)
if h:
    buf=ctypes.create_unicode_buffer(32768);size=wintypes.DWORD(len(buf))
    assert k.QueryFullProcessImageNameW(h,0,buf,ctypes.byref(size))
    assert Path(buf.value).resolve()==Path(sys.executable).resolve(),'Unexpected reused process identity'
    while k.WaitForSingleObject(h,0)==258:
        write(OUT/'pipeline_riesgo_estado.json',dict(stage='Waiting for actual preceding GPU/control chain exit',wait_original_controller=34908,supervisor_pid=os.getpid(),goal='active',Test='sealed'))
        time.sleep(15)
    k.CloseHandle(h)
parents=[OUT/'prefijo_nativo_s0']
if not (parents[0]/'resultado.json').exists():
    write(OUT/'pipeline_riesgo_estado.json',dict(stage='Preceding native result incomplete; inspect and recover before any risk training',goal='active',Test='sealed'));exit()
pilot_path=OUT/'piloto_historia_condicional_s0/resultado.json'
if not pilot_path.exists():
    write(OUT/'pipeline_riesgo_estado.json',dict(stage='Conditional CPU pilot incomplete; inspect its log before continuing',goal='active',Test='sealed'));exit()
pilot=read(pilot_path)
if pilot['worth_GPU_test']:
    hp=OUT/'prefijo_historia_s0'
    if not (hp/'resultado.json').exists():
        write(OUT/'pipeline_riesgo_estado.json',dict(stage='History GPU stage incomplete; inspect and recover it before a different objective',goal='active',Test='sealed'));exit()
    parents.append(hp)
records=[];keep=[];eligible_target=False
for parent in parents:
    r=read(parent/'resultado.json')
    assert all(m['n']==127941 for m in r['all_alpha_validation'])
    assert sha(parent/'modelo.pt')==r['model_sha256']
    records.append(dict(id=parent.name,validation=r['validation'],graph_validation=r['graph_validation'],selected_alpha=r['selected_alpha'],all_outputs=r['all_alpha_validation'],paired=r['paired_vs_retained'],model_sha256=r['model_sha256']))
    eligible_target|=any(m['mae_s']<73.5 for m in r['all_alpha_validation'])
    for a,m in zip(r['config']['alphas'],r['all_alpha_validation']):
        if a>0 and m['mae_s']<131.25671207822356:
            keep.append(dict(id=parent.name,alpha=a,validation=m,checkpoint=str(parent/'modelo.pt'),model_sha256=r['model_sha256'],reference_sha256=r['reference_sha256'],threshold_qualifies=m['mae_s']<131.06,reproduction_pending=True))
write(OUT/'candidatos_conservados_antes_riesgo.json',dict(candidates=keep,protected_original_model=True,TransferTraj_unchanged=True,Test='sealed'))
write(OUT/'conclusion_antes_riesgo_total.json',dict(complete_models=records,conditional_head_pilot=pilot,decision='Previous full results recorded before a new objective. Teacher-forcing likelihood does not optimize the marginal travel-time distribution directly; test conditional full-trip CRPS with teacher regularization.',scope='A proposed experiment, no claim it will reach73.5s. All queries and labels unchanged; alpha0 is no learned improvement.',Test='sealed',goal='active'))
print(json.dumps(dict(stage='preceding_results_complete',models=records,candidates=keep)),flush=True)
if eligible_target:
    write(OUT/'pipeline_riesgo_estado.json',dict(stage='Validation target candidate must be reproduced, frozen and confirmed; no more training launched',goal='active',Test='sealed'));exit()
assert read(OUT/'verificacion_muestreador_rechazo.json')['passed']
assert read(OUT/'sondeo_riesgo_total_cpu.json')['passed']
run=OUT/'riesgo_total_s0'
assert not run.exists(),'Never overwrite or duplicate an existing trial'
log=(OUT/'riesgo_total_s0.log').open('w',encoding='utf-8')
p=subprocess.Popen([sys.executable,'-B',str(OUT/'entrenar_riesgo_total.py'),'--id','riesgo_total_s0'],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
active=dict(experiment='riesgo_total_s0',pid=p.pid,device='GPU',supervisor_pid=os.getpid(),max_GPU_jobs=1,objective='Conditional full-trip CRPS on generated complete paths plus teacher regularization',GPU_gradient_gate_required=True,do_not_modify_running_sources=True)
for name in ['sesiones_activas.json','sesiones_extra.json']:write(OUT/name,dict(active=[active],goal='active',Test='sealed'))
c=read(OUT/'CONTINUIDAD_OPERATIVA.json');c['current_trial']=active;c['latest_complete_models_before_risk']=records;write(OUT/'CONTINUIDAD_OPERATIVA.json',c)
print(json.dumps(dict(stage='risk_GPU_started',**active)),flush=True)
while p.poll() is None:
    write(OUT/'pipeline_riesgo_estado.json',dict(stage='Full-trip risk trial active; read log and mandatory GPU gate',active=active,goal='active',Test='sealed'))
    time.sleep(15)
log.close()
complete=(run/'resultado.json').exists()
if complete:
    r=read(run/'resultado.json')
    qualified=[]
    for a,m in zip(r['config']['alphas'],r['all_alpha_validation']):
        if a>0 and m['mae_s']<131.25671207822356:
            qualified.append(dict(id=run.name,alpha=a,validation=m,model_sha256=r['model_sha256'],threshold_qualifies=m['mae_s']<131.06,reproduction_and_MC_robustness_pending=True))
    write(OUT/'candidatos_conservados_riesgo_total.json',dict(candidates=qualified,keep_weights=bool(qualified),original_reference_protected=True,Test='sealed'))
    write(OUT/'conclusion_riesgo_total.json',dict(validation=r['validation'],graph=r['graph_validation'],all_outputs=r['all_alpha_validation'],best_step=r['best_step'],learned_update_selected=r['learned_update_selected'],paired=r['paired_vs_retained'],support_failure_rate=r['support_failure_rate'],goal='active',Test='sealed',next='If target eligible, reproduce/freeze/confirm; otherwise draw a critical conclusion and pursue a different improvement, never stop goal merely because this trial ended'))
write(OUT/'pipeline_riesgo_estado.json',dict(stage='Inspect complete risk result and continue goal' if complete else 'Risk stage ended without result; inspect actual log and checkpoint, do not overwrite',exit_code=p.returncode,complete_result=complete,goal='active',Test='sealed'))
print(json.dumps(dict(stage='risk_stage_ended',exit_code=p.returncode,complete_result=complete,goal='active')),flush=True)
