"""Reproduce the retained best checkpoint from its frozen sources; never trains or opens Test labels."""
from pathlib import Path
import os,sys,json,hashlib,types,time,argparse
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'propuesta'
BEST_ID='od_mezcla8_geometria_conjunta_s0_fp32_v2'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='propuesta/instrumentos/verificacion_modelo_conservado.json');args=parser.parse_args();destination=(ROOT/args.output).resolve();assert destination.is_relative_to(ROOT)
    assert Path(sys.executable).resolve().is_relative_to(ROOT/'.conda-env');started=time.time();run=BASE/'experimentos'/BEST_ID;frozen=run/'fuentes';record=json.loads((run/'resultado.json').read_text(encoding='utf-8-sig'));execution=json.loads((run/'ejecucion.json').read_text(encoding='utf-8-sig'))
    for name,h in execution['code_hashes'].items():assert sha(frozen/name)==h,name
    assert sha(run/'modelo.pt')==record['model_sha256']
    assert sha(BASE/'datos/manifiesto.json')==record['data_manifest_sha256']
    for name,h in execution['input_hashes'].items():assert sha(ROOT/name)==h,name
    accessed=set()
    def audit(event,arguments):
        if event!='open' or not arguments or not isinstance(arguments[0],(str,bytes,os.PathLike)):return
        try:p=Path(os.fsdecode(arguments[0])).resolve()
        except (OSError,ValueError):return
        if p.is_relative_to(BASE/'datos'):accessed.add(str(p.relative_to(ROOT)))
    sys.addaudithook(audit)
    sys.path=[str(frozen)]+[v for v in sys.path if Path(v or os.getcwd()).resolve()!=BASE];common=types.ModuleType('common');common.__file__=str(BASE/'common.py');sys.modules['common']=common;exec(compile((frozen/'common.py').read_text(encoding='utf-8-sig'),common.__file__,'exec'),common.__dict__)
    import numpy as np,torch,socket
    def deny(*a,**kw):raise RuntimeError('No network access during best-model reproduction')
    socket.socket.connect=deny;socket.create_connection=deny;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from inferencia import predict_run
    metadata,prediction=predict_run(run,'validation');saved=np.load(run/'validacion.npz');assert np.array_equal(metadata['row'],saved['row']);delta=float(np.max(np.abs(prediction-saved['pred'])));assert delta==0.,delta
    mae=float(np.abs(saved['true'].astype(float)-prediction.astype(float)).mean());assert abs(mae-record['validation']['mae_s'])<1e-10
    from predecir_od import predict_new_od
    fields=['o_lon','o_lat','d_lon','d_lat','timestamp'];items=[{name:metadata[name][i].item() for name in fields} for i in range(65)];raw,_,_=predict_new_od(run,items);raw_delta=float(np.max(np.abs(raw-prediction[:65])));assert raw_delta==0.,raw_delta
    reverse,_,_=predict_new_od(run,items[::-1]);single,_,_=predict_new_od(run,items[:1]);reverse_delta=float(np.max(np.abs(reverse[::-1]-raw)));single_delta=float(np.max(np.abs(single-raw[:1])));assert reverse_delta==0. and single_delta==0.
    violations=[]
    for name,module in list(sys.modules.items()):
        source=getattr(module,'__file__',None)
        if source and name not in ['__main__','__mp_main__','common']:
            p=Path(source).resolve()
            if p.is_relative_to(BASE) and not p.is_relative_to(frozen):violations.append((name,str(p)))
    assert not violations,violations
    proof=dict(passed=True,model_id=BEST_ID,model_sha256=record['model_sha256'],validation_n=len(prediction),validation_mae_s=mae,all_validation_predictions_bit_identical=True,validation_max_delta_s=delta,raw65_max_delta_s=raw_delta,reverse_max_delta_s=reverse_delta,single_max_delta_s=single_delta,frozen_sources_verified=len(execution['code_hashes']),current_research_imports=0,actual_data_files_accessed=sorted(accessed),no_training=True,no_Test_labels_read=True,seconds=time.time()-started)
    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(proof),flush=True)

if __name__=='__main__':main()
