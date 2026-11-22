from common import *
import subprocess,ctypes
# Local, bounded experiment stages. Selection remains validation-only.
state=dict(started_unix=time.time(),root=str(ROOT),jobs=[],test_labels_read=False)
def save():write_json(BASE/'cola_estado.json',state)
def free_gpu():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],cwd=ROOT,text=True)
    return int(out.strip().splitlines()[0])
def execute(name,script,args,dependencies=(),minimum_free=3000):
    if (RUNS/name/'resultado.json').exists():return json.loads((RUNS/name/'resultado.json').read_text(encoding='utf-8-sig'))
    rec=dict(id=name,script=script,args=args,status='waiting',dependencies=[str(p.relative_to(ROOT)) for p in dependencies],created_unix=time.time());state['jobs'].append(rec);save()
    while not all(p.exists() for p in dependencies) or free_gpu()<minimum_free:
        time.sleep(20)
    rec.update(status='running',started_unix=time.time());save();log=BASE/('cola_'+name+'.log')
    with log.open('x',encoding='utf-8') as output:
        process=subprocess.Popen([sys.executable,str(BASE/script),*args],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,creationflags=0x08000000 if os.name=='nt' else 0)
        rec['pid']=process.pid;save();code=process.wait()
    rec.update(returncode=code,finished_unix=time.time(),status='completed' if code==0 else 'failed');save()
    if code!=0:raise RuntimeError(f'{name} failed; see {log}')
    for report in ('actualizar_bitacora.py','explorar_ensambles.py','actualizar_estado.py'):
        subprocess.run([sys.executable,str(BASE/report)],cwd=ROOT,stdout=subprocess.DEVNULL,check=True,creationflags=0x08000000 if os.name=='nt' else 0)
    return json.loads((RUNS/name/'resultado.json').read_text(encoding='utf-8-sig'))
# Prevent automatic system sleep only while this process is active; display can sleep.
if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
try:
    first='od_xgboost_mapa2014_s0';r14=execute(first,'experimento_comparadores.py',['--id',first,'--engine','xgboost','--archive-maps','2014'],[DATA/'mapa_2014_features.json'])
    suffix='od_sufijos_sin_adaptador_s0';execute(suffix,'experimento_sufijos.py',['--id',suffix,'--no-adapter','--pretrained',str(RUNS/'transformer_preentrenamiento5_s0/epoca_5.pt')],minimum_free=2200)
    second='od_xgboost_mapa2026_s0';r26=execute(second,'experimento_comparadores.py',['--id',second,'--engine','xgboost','--archive-maps','2026'],[DATA/'mapa_2026_features.json'])
    baseline=json.loads((RUNS/'od_xgboost_contexto_rutas_s0/resultado.json').read_text(encoding='utf-8-sig'));best=min((r14,r26),key=lambda r:r['validation']['mae_s']);gain=baseline['validation']['mae_s']-best['validation']['mae_s'];state['map_decision']=dict(best=best['id'],gain_s=gain,rule='Joint-map experiment if gain >=0.3s; TabM map transfer if >=1s. Practical stage rules, not statistical significance.');save()
    if gain>=.3:
        name='od_xgboost_mapas2014_2026_s0';execute(name,'experimento_comparadores.py',['--id',name,'--engine','xgboost','--archive-maps','2014','2026'])
    else:state['joint_map_skipped']='Neither single map improved XGBoost by 0.3s';save()
    if gain>=1:
        year='2014' if best['id']==first else '2026';name='od_tabm_mapa'+year+'_s0';execute(name,'experimento_tabm.py',['--id',name,'--archive-maps',year])
    seed='od_tabm_geometria_rutas_s1';execute(seed,'experimento_tabm.py',['--id',seed,'--map-geometry','--seed','1'])
    state['status']='stage_queue_completed';save()
finally:
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
