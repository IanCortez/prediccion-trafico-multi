from common import *
import subprocess,ctypes
name='transformer_preentrenamiento30_s0';source=RUNS/'transformer_preentrenamiento5_s0/epoca_5.pt';state=dict(started_unix=time.time(),status='running',jobs=[],test_labels_read=False)
def save():write_json(BASE/'fase_pre30_estado.json',state)
def execute(identifier,script,args,result_file='resultado.json',create_run=False):
    run=RUNS/identifier;assert not run.exists(),identifier
    if create_run:run.mkdir()
    job=dict(id=identifier,script=script,args=args,started_unix=time.time(),status='running');state['jobs'].append(job);save()
    with (BASE/('pre30_'+identifier+'.log')).open('x',encoding='utf-8') as out:
        process=subprocess.Popen([sys.executable,str(BASE/script),*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,creationflags=0x08000000);job['pid']=process.pid;save();code=process.wait()
    job.update(returncode=code,finished_unix=time.time(),status='completed' if code==0 else 'failed');save();assert code==0,identifier;assert (run/result_file).exists();return json.loads((run/result_file).read_text())
if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
try:
    previous=json.loads((source.parent/'resultado_preentrenamiento.json').read_text());assert previous['epochs']==5 and previous['n_trips']==1023485 and previous['only_train'];assert source.exists()
    plan=dict(created_unix=time.time(),source=str(source.relative_to(ROOT)),source_sha256=sha(source),previous_epochs=5,total_epochs=30,additional_epochs=25,train_trips=previous['n_trips'],train_points=previous['points'],same_optimizer_and_rng_state_resumed=True,finetunes=[dict(id='od_transformer_pre30_s0',context='counts',reference='od_transformer_pre5_s0'),dict(id='od_transformer_pre30_semantica_s0',context='semantic',reference='od_transformer_pre5_semantica_s0')],hypothesis='Test whether the previous five-epoch pretraining budget understated the value of trajectory reconstruction. Thirty epochs matches the paper budget in name only; this remains our architecture/cohort, not a TransferTraj replication.',test_labels_read=False);write_json(BASE/'instrumentos/plan_preentrenamiento30.json',plan)
    execute(name,'preentrenar_trayectorias.py',['--id',name,'--epochs','30','--resume',str(source)],'resultado_preentrenamiento.json',True)
    for config in plan['finetunes']:
        execute(config['id'],'experimento_transformer.py',['--id',config['id'],'--pretrained',str(RUNS/name/'epoca_30.pt'),'--context-grid',config['context']])
        for script in ('actualizar_bitacora.py','explorar_ensambles.py','actualizar_estado.py'):
            subprocess.run([sys.executable,str(BASE/script)],cwd=ROOT,stdout=subprocess.DEVNULL,check=True,creationflags=0x08000000)
    state.update(status='stage_completed',finished_unix=time.time());save()
except Exception as error:
    state.update(status='failed',error=repr(error),finished_unix=time.time());save();raise
finally:
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
