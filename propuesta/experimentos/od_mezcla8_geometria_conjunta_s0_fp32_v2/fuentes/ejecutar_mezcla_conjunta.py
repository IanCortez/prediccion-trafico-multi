from common import *
import subprocess,ctypes
state=dict(started_unix=time.time(),status='running',jobs=[],test_labels_read=False)
def save():write_json(BASE/'fase_conjunta_estado.json',state)
def wait_for(predicate,description):
    started=time.time();state['waiting_for']=description;save()
    while not predicate():
        if time.time()-started>14400:raise TimeoutError(description)
        time.sleep(15)
    state.pop('waiting_for',None);save()
def gpu_free():return int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],cwd=ROOT,text=True).strip().splitlines()[0])
if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
try:
    for name in ('mezcla_conjunta_contrato.json','mezcla_conjunta_forward.json'):assert json.loads((BASE/'instrumentos'/name).read_text())['passed']
    plan=dict(created_unix=time.time(),hypothesis='A shared latent component can learn duration and route geometry from training, avoiding hard assignment to an incomplete static candidate set.',configurations=[dict(id='od_mezcla8_sin_geometria_s0',geometry_weight=0.),dict(id='od_mezcla8_geometria_conjunta_s0',geometry_weight=.2)],common=dict(seed=0,components=8,geometry_dim=14,max_epochs=100,early_stopping=20,initial_lr=.001,batch=512,k=16,width=512,features='Same216 OD/map/calendar/history features as the existing vial mixture.'),loss='NLL_time +5*RPS +lambda*NLL(geometry|observed training duration,OD)/14. Component marginalization precedes division by14.',inference='Marginal time distribution from OD only. No observed query route or duration.',caveat='Latent component geometry means are not guaranteed valid road paths; mixture labels are not identifiable route classes.',test_labels_read=False,code_hashes={name:sha(BASE/name) for name in ('modelo_mezcla_conjunta.py','experimento_mezcla_conjunta.py')});write_json(BASE/'instrumentos/plan_mezcla_conjunta.json',plan);processes=[]
    for cfg in plan['configurations']:
        name=cfg['id'];assert not (RUNS/name).exists();wait_for(lambda:gpu_free()>=4500,'GPU memory for '+name);out=(BASE/('conjunta_'+name+'.log')).open('x',encoding='utf-8');args=[sys.executable,str(BASE/'experimento_mezcla_conjunta.py'),'--id',name,'--geometry-weight',str(cfg['geometry_weight'])];proc=subprocess.Popen(args,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,creationflags=0x08000000);job=dict(id=name,pid=proc.pid,status='running',started_unix=time.time());state['jobs'].append(job);save();processes.append((proc,out,job));wait_for(lambda:(RUNS/name/'modelo.pt').exists() or proc.poll() is not None,'First checkpoint and GPU allocation '+name)
        if proc.poll() not in (None,0):raise RuntimeError(name+' failed before first checkpoint')
    for proc,out,job in processes:
        code=proc.wait();out.close();job.update(status='completed' if code==0 else 'failed',returncode=code,finished_unix=time.time());save()
    assert all(job['returncode']==0 for _,_,job in processes)
    for _,_,job in processes:
        source=job['id'];target=source+'_fp32_v2';assert not (RUNS/target).exists()
        with (BASE/('conjunta_auditoria_'+source+'.log')).open('x',encoding='utf-8') as out:subprocess.run([sys.executable,str(BASE/'auditar_precision.py'),'--ids',source],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True,creationflags=0x08000000)
    from evaluar_iteracion import report_iteration
    for _,_,job in processes:report_iteration(json.loads((RUNS/job['id']/'resultado.json').read_text()))
    for script in ('verificar_inferencia.py','verificar_consulta_od.py','comparar_fase_vial.py','actualizar_bitacora.py','explorar_ensambles.py','actualizar_estado.py'):
        with (BASE/('conjunta_verificacion_'+script+'.log')).open('w',encoding='utf-8') as out:subprocess.run([sys.executable,str(BASE/script)],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True,creationflags=0x08000000)
    state.update(status='stage_completed',finished_unix=time.time());save()
except Exception as error:
    state.update(status='failed',error=repr(error),finished_unix=time.time());save();raise
finally:
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
