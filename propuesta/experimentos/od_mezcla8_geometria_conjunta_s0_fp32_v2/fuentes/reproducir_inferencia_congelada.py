"""Replay validation inference with the exact Python sources copied at run start.
A fresh worker prevents imports from another run from contaminating the replay.
"""
from pathlib import Path
import sys,os,json,hashlib,types,subprocess,argparse,time
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'propuesta';RUNS=BASE/'experimentos'
assert Path(sys.executable).resolve().is_relative_to(ROOT)
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
p=argparse.ArgumentParser();p.add_argument('--ids',nargs='+');p.add_argument('--worker');a=p.parse_args()
if a.worker:
    run=(RUNS/a.worker).resolve();assert run.is_relative_to(RUNS);folder=run/'fuentes';execution=json.loads((run/'ejecucion.json').read_text(encoding='utf-8-sig'));hashes=execution['code_hashes']
    for name,value in hashes.items():assert digest(folder/name)==value,(name,'Frozen source hash mismatch')
    assert 'common.py' in hashes and 'inferencia.py' in hashes
    sys.path.insert(0,str(folder))
    # The historical common module resolves the project root from __file__.
    # Give that one module its original project location; execute its frozen bytes.
    common=types.ModuleType('common');common.__file__=str(BASE/'common.py');sys.modules['common']=common;exec(compile((folder/'common.py').read_text(encoding='utf-8-sig'),common.__file__,'exec'),common.__dict__);assert common.ROOT==ROOT
    from inferencia import predict_run
    import numpy as np
    d,pred=predict_run(run,'validation');z=np.load(run/'validacion.npz');assert np.array_equal(d['row'],z['row']);delta=float(np.max(np.abs(pred-z['pred'])));assert delta<.05,(a.worker,delta)
    current_imports=[]
    for name,module in list(sys.modules.items()):
        value=getattr(module,'__file__',None)
        if value and name not in ('__main__','__mp_main__','common'):
            path=Path(value).resolve()
            if path.is_relative_to(BASE) and not path.is_relative_to(folder):current_imports.append((name,str(path)))
    assert not current_imports,('Replay imported current research modules',current_imports)
    print(json.dumps(dict(id=a.worker,n=len(pred),max_prediction_change_s=delta,frozen_python_files_verified=len(hashes),frozen_inference_sha256=hashes['inferencia.py'],current_research_module_imports=0,test_labels_read=False,root_resolution_override_only=True)),flush=True)
else:
    names=a.ids or ['od_mezcla_proyectada_mm_calendario_s0_fp32_v2','od_transformer_pre30_semantica_s0_fp32_v2'];out=[]
    for name in names:
        proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',name],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',creationflags=0x08000000 if os.name=='nt' else 0);(BASE/'instrumentos'/('fuentes_replay_'+name+'.log')).write_text(proc.stdout+proc.stderr,encoding='utf-8');assert proc.returncode==0,(name,proc.stdout[-1500:],proc.stderr[-1500:]);record=json.loads(proc.stdout.strip().splitlines()[-1]);out.append(record);print(record,flush=True)
    (BASE/'instrumentos/fuentes_congeladas_inferencia.json').write_text(json.dumps(dict(passed=True,models=out,verified_unix=time.time(),method='Each fresh worker verifies copied source hashes, executes frozen common with its original root location, and imports only frozen research modules. Third-party dependencies remain the pinned local environment.',test_labels_read=False),indent=2),encoding='utf-8')
