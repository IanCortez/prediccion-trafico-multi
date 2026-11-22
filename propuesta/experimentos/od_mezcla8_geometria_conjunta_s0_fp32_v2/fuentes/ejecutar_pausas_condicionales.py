from common import *
import subprocess,ctypes
state=dict(started_unix=time.time(),status='waiting',jobs=[],test_labels_read=False)
def save():write_json(BASE/'fase_pausas_estado.json',state)
def wait_for(predicate,description):
    started=time.time();state['waiting_for']=description;save()
    while not predicate():
        if time.time()-started>14400:raise TimeoutError(description)
        time.sleep(15)
    state.pop('waiting_for',None);save()
def previous_stage():
    try:return json.loads((BASE/'fase_conjunta_estado.json').read_text())['status']=='stage_completed'
    except (FileNotFoundError,json.JSONDecodeError):return False
def execute(identifier,script,args,min_free=4500):
    assert not (RUNS/identifier).exists();wait_for(lambda:int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],cwd=ROOT,text=True).strip().splitlines()[0])>=min_free,'GPU memory for '+identifier);job=dict(id=identifier,status='running',script=script,args=args,started_unix=time.time());state['jobs'].append(job);state['status']='running';save()
    with (BASE/('pausas_'+identifier+'.log')).open('x',encoding='utf-8') as out:
        proc=subprocess.Popen([sys.executable,str(BASE/script),*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,creationflags=0x08000000);job['pid']=proc.pid;save();code=proc.wait()
    job.update(status='completed' if code==0 else 'failed',returncode=code,finished_unix=time.time());save();assert code==0,identifier
if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
try:
    for name in ('pausas_beta_binomial_contrato.json','mezcla_conjunta_forward_pausas.json'):assert json.loads((BASE/'instrumentos'/name).read_text())['passed']
    diagnostic=json.loads((BASE/'instrumentos/diagnostico_error_pausas.json').read_text());share=sum(row['error_share'] for row in diagnostic['strata']['low_displacement5m_seconds'] if row['interval'][0]>=180);assert share>=.25
    plan=dict(created_unix=time.time(),hypothesis='Operational GPS low-displacement counts may help latent duration components represent long-delay heterogeneity beyond shape alone.',id='od_mezcla8_pausas_condicionales_s0',reference='od_mezcla8_sin_geometria_s0',common='Same216 OD features, eight components, seed0, architecture/initialization/optimizer and max100 epochs as existing zero-auxiliary control.',geometry_weight=0,pause_weight=.2,pause_threshold_m=5,train_target_sha256=sha(DATA/'pausas_gps_train.npz'),diagnostic_error_share_long_low_displacement=share,loss='Time NLL +5RPS +0.2*NLL(S|T,OD), with beta-binomial count S and n=T/15 in the training likelihood only.',inference='Time marginal from OD only; observed point count, duration and low-displacement count are not inputs.',stage_gate='Run after the current joint phase, if its zero-auxiliary control is within2s of the best completed primary individual.',test_labels_read=False,code_hashes={name:sha(BASE/name) for name in ('modelo_mezcla_conjunta.py','experimento_mezcla_conjunta.py')});write_json(BASE/'instrumentos/plan_pausas_condicionales.json',plan)
    wait_for(previous_stage,'Current joint-geometry phase and its audits');reference=json.loads((RUNS/(plan['reference']+'_fp32_v2')/'resultado.json').read_text());from evaluar_iteracion import eligible,read_records;best=min(r['validation']['mae_s'] for r in read_records() if eligible(r));state['gate']=dict(reference_mae_s=reference['validation']['mae_s'],best_primary_mae_s=best,allowed=reference['validation']['mae_s']<=best+2);save()
    if state['gate']['allowed']:
        name=plan['id'];execute(name,'experimento_mezcla_conjunta.py',['--id',name,'--geometry-weight','0','--pause-weight','.2']);execute(name+'_fp32_v2','auditar_precision.py',['--ids',name],2500)
        for script in ('verificar_inferencia.py','verificar_consulta_od.py','comparar_fase_vial.py','actualizar_bitacora.py','explorar_ensambles.py','actualizar_estado.py'):
            with (BASE/('pausas_verificacion_'+script+'.log')).open('w',encoding='utf-8') as out:subprocess.run([sys.executable,str(BASE/script)],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True,creationflags=0x08000000)
    else:state['decision']='Skip: current zero-auxiliary backbone is more than2s behind the best completed primary model.'
    state.update(status='stage_completed',finished_unix=time.time());save()
except Exception as error:
    state.update(status='failed',error=repr(error),finished_unix=time.time());save();raise
finally:
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
