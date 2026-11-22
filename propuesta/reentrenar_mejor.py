"""Explicit optional launcher for the retained training recipe. Not run by cleanup."""
from pathlib import Path
import argparse,sys,os,json,hashlib,types,shutil
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'propuesta';BEST='od_mezcla8_geometria_conjunta_s0_fp32_v2'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--id',required=True,help='New run identifier; must not exist');args=parser.parse_args();run=(BASE/'experimentos'/args.id).resolve();assert run.parent==BASE/'experimentos' and not run.exists();assert Path(sys.executable).resolve().is_relative_to(ROOT/'.conda-env')
    source=BASE/'experimentos'/BEST;frozen=source/'fuentes';execution=json.loads((source/'ejecucion.json').read_text())
    for name,h in execution['code_hashes'].items():assert digest(frozen/name)==h,name
    for name,h in execution['input_hashes'].items():assert digest(ROOT/name)==h,name
    data_manifest=json.loads((BASE/'datos/manifiesto.json').read_text());record=json.loads((source/'resultado.json').read_text());assert digest(BASE/'datos/manifiesto.json')==record['data_manifest_sha256']
    for split in ['train','validation']:
        assert digest(BASE/'datos'/(split+'_x.npz'))==data_manifest[split]['x_sha256']
        assert digest(BASE/'datos'/(split+'_y.npy'))==data_manifest[split]['y_sha256']
    sys.path=[str(frozen)]+[s for s in sys.path if Path(s or os.getcwd()).resolve()!=BASE];common=types.ModuleType('common');common.__file__=str(BASE/'common.py');sys.modules['common']=common;exec(compile((frozen/'common.py').read_text(encoding='utf-8-sig'),common.__file__,'exec'),common.__dict__)
    original_snapshot=common.snapshot
    def snapshot(destination,config):
        original_snapshot(destination,config)
        for name in execution['code_hashes']:shutil.copy2(frozen/name,destination/'fuentes'/name)
        p=destination/'ejecucion.json';saved=json.loads(p.read_text());saved['code_hashes']={p.name:digest(p) for p in (destination/'fuentes').glob('*.py')};saved['retained_training_recipe_origin']=BEST;common.write_json(p,saved)
    common.snapshot=snapshot
    # Reporting is limited to the new run; discarded historical branches are not launched.
    common.register=lambda record:common.write_json(run/'registro_reentrenamiento.json',record)
    sys.argv=[str(frozen/'experimento_mezcla_conjunta.py'),'--id',args.id,'--geometry-weight','0.2','--seed','0','--epochs','100','--batch','512','--k','16','--width','512']
    scope={'__name__':'__main__','__file__':str(frozen/'experimento_mezcla_conjunta.py')};exec(compile((frozen/'experimento_mezcla_conjunta.py').read_text(encoding='utf-8-sig'),scope['__file__'],'exec'),scope)

if __name__=='__main__':main()
