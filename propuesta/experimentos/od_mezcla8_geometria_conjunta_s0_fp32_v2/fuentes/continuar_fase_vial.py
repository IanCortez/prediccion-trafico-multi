from common import *
import subprocess,ctypes
state=dict(started_unix=time.time(),status='running',jobs=[],test_labels_read=False)
def save():write_json(BASE/'fase_vial_estado.json',state)
def gpu_free():return int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],cwd=ROOT,text=True).strip().splitlines()[0])
def wait_for(predicate,description):
    started=time.time();state['waiting_for']=description;save()
    while not predicate():
        if time.time()-started>10800:raise TimeoutError(description)
        time.sleep(15)
    state.pop('waiting_for',None);save()
def report():
    for name in ('actualizar_bitacora.py','explorar_ensambles.py','actualizar_estado.py'):
        subprocess.run([sys.executable,str(BASE/name)],cwd=ROOT,stdout=subprocess.DEVNULL,check=True,creationflags=0x08000000)
def execute(name,script,args,min_free=2600):
    if (RUNS/name/'resultado.json').exists():return
    assert not (RUNS/name).exists(),f'Existing incomplete run requires inspection: {name}'
    wait_for(lambda:gpu_free()>=min_free,'GPU memory for '+name);job=dict(id=name,script=script,args=args,started_unix=time.time(),status='running');state['jobs'].append(job);save()
    with (BASE/('fase_'+name+'.log')).open('x',encoding='utf-8') as out:
        proc=subprocess.Popen([sys.executable,str(BASE/script),*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,creationflags=0x08000000);job['pid']=proc.pid;save();code=proc.wait()
    job.update(status='completed' if code==0 else 'failed',returncode=code,finished_unix=time.time());save()
    if code:raise RuntimeError(name+' failed')
    report()
if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
try:
    wait_for(lambda:(RUNS/'od_xgboost_ruta_preferida_s0/resultado.json').exists(),'Existing learned-route XGBoost experiment')
    gain=json.loads((BASE/'instrumentos/cobertura_ruta_preferida.json').read_text())['oracle_coverage_gain'];xgb=json.loads((RUNS/'od_xgboost_ruta_preferida_s0/resultado.json').read_text());parent=json.loads((RUNS/'od_xgboost_movimiento_mapmatched_s0/resultado.json').read_text());state['route_decision']=dict(xgboost_gain_s=parent['validation']['mae_s']-xgb['validation']['mae_s'],diagnostic_oracle_coverage_gain=gain,rule='One matched TabM comparison if XGBoost gain>=0.2s or candidate coverage gain>=2pp. Diagnostic coverage is not an inference input.');save();report()
    for source in ('od_tabm_geometria_rutas_s1','od_tabm_proyectado_mm_calendario_s0'):
        execute(source+'_fp32_v2','auditar_precision.py',['--ids',source],2200)
    if state['route_decision']['xgboost_gain_s']>=.2 or gain>=.02:
        name='od_tabm_ruta_preferida_s0';execute(name,'experimento_tabm.py',['--id',name,'--mapmatched','--calendar','--learned-route'],3400)
        execute(name+'_fp32_v2','auditar_precision.py',['--ids',name],2200)
    for source in ('od_mezcla_proyectada_mm_calendario_s0','od_mezcla_rutas_sin_aux_s1'):
        wait_for(lambda:(RUNS/source/'resultado.json').exists(),'Existing mixture '+source)
        execute(source+'_fp32_v2','auditar_precision.py',['--ids',source],2300)
    report();state.update(status='stage_completed',finished_unix=time.time());save()
except Exception as error:
    state.update(status='failed',error=repr(error),finished_unix=time.time());save();raise
finally:
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
