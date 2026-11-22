from common import *
import subprocess,ctypes
source='od_mezcla_proyectada_mm_calendario_s0_fp32_v2';state=dict(started_unix=time.time(),status='running',jobs=[],test_labels_read=False);path=BASE/'fase_mediana_estado.json'
def save():write_json(path,state)
def wait_for(predicate,description):
    started=time.time();state['waiting_for']=description;save()
    while not predicate():
        if time.time()-started>10800:raise TimeoutError(description)
        time.sleep(15)
    state.pop('waiting_for',None);save()
def gpu_free():return int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],cwd=ROOT,text=True).strip().splitlines()[0])
if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
try:
    wait_for(lambda:(RUNS/source/'resultado.json').exists(),'Frozen float32 source checkpoint');parent=json.loads((RUNS/source/'resultado.json').read_text());plan=dict(source=source,source_model_sha256=parent['model_sha256'],source_record_sha256=sha(RUNS/source/'resultado.json'),configurations=[dict(id='od_mezcla_vial_continuacion_s0',median_l1_weight=0),dict(id='od_mezcla_vial_mediana_s0',median_l1_weight=3)],common=dict(seed=0,max_additional_epochs=30,initial_lr=.0001,weight_decay=.0003,batch=512,early_stopping=8,median_mode='interpolated',inference_precision='float32',canonical_batch=2048),loss='NLL +5*mean_squared_CDF_error +weight*mean(abs(interpolated_median-y))/600',hypothesis='A targeted median loss can improve MAE beyond identical additional optimization of NLL/RPS.',test_labels_read=False,created_unix=time.time());write_json(BASE/'instrumentos/plan_mediana_dirigida.json',plan);processes=[]
    for config in plan['configurations']:
        name=config['id'];assert not (RUNS/name).exists();wait_for(lambda:gpu_free()>=3500,'GPU memory for '+name);output=(BASE/('mediana_'+name+'.log')).open('x',encoding='utf-8');args=[sys.executable,str(BASE/'experimento_mediana_dirigida.py'),'--id',name,'--source',source,'--pinball',str(config['median_l1_weight'])];proc=subprocess.Popen(args,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,creationflags=0x08000000);job=dict(id=name,pid=proc.pid,started_unix=time.time(),status='running');state['jobs'].append(job);save();processes.append((proc,output,job));wait_for(lambda:(RUNS/name/'modelo.pt').exists() or proc.poll() is not None,'GPU allocation and initial prediction '+name)
        if proc.poll() not in (None,0):raise RuntimeError(name+' initialization failed')
    for proc,output,job in processes:
        code=proc.wait();output.close();job.update(status='completed' if code==0 else 'failed',returncode=code,finished_unix=time.time());save()
    assert all(job['returncode']==0 for _,_,job in processes)
    from evaluar_iteracion import report_iteration
    for _,_,job in processes:report_iteration(json.loads((RUNS/job['id']/'resultado.json').read_text()))
    for script in ('verificar_inferencia.py','verificar_consulta_od.py','actualizar_bitacora.py','explorar_ensambles.py','actualizar_estado.py'):
        with (BASE/('mediana_verificacion_'+script+'.log')).open('w',encoding='utf-8') as out:subprocess.run([sys.executable,str(BASE/script)],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True,creationflags=0x08000000)
    state.update(status='stage_completed',finished_unix=time.time());save()
except Exception as error:
    state.update(status='failed',error=repr(error),finished_unix=time.time());save();raise
finally:
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
