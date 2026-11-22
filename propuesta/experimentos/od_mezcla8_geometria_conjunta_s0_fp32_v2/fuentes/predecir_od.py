from common import *
from consulta_od import od_records,build_od_features
import torch
from modelo_tabm import CityTabM

def predict_new_od(run,items):
    run=Path(run);record=json.loads((run/'resultado.json').read_text(encoding='utf-8-sig'));assert record.get('cohort','primary')=='primary','Privileged/diagnostic model is ineligible'
    if record['architecture']=='trajectory_transformer':
        from modelo_trayectorias import TrajectoryEncoder
        from features import features
        path=run/'modelo.pt';assert sha(path)==record['model_sha256'];d=od_records(items);F,names=features(d);ck=torch.load(path,map_location='cpu',weights_only=False);torch.set_num_threads(4);torch.set_float32_matmul_precision('highest' if record['config'].get('inference_precision')=='float32' else 'high');m=TrajectoryEncoder(ck['state']['context_grid'],ck['grid_info'],ck['coef']).cuda();m.load_state_dict(ck['state']);m.eval();xy=np.stack([F[:,:2],F[:,2:4]],axis=1);out=[]
        for start in range(0,len(F),8192):
            points=torch.as_tensor(xy[start:start+8192],device='cuda');dep=torch.as_tensor(F[start:start+8192,18:24],device='cuda');known=torch.zeros(points.shape[:2],device='cuda');visible=torch.zeros_like(known,dtype=torch.bool);visible[:,0]=True
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=record['config'].get('inference_precision')!='float32'):pred=m(points,known,visible,dep)[0][:,1]
            out.append(pred.cpu().numpy())
        return np.maximum(np.concatenate(out),1),F,names
    assert record['architecture'] in ('tabm','tabm_route_mixture','tabm_joint_route_time'),'This raw-query adapter supports TabM and its duration mixture'
    path=run/'modelo.pt';assert sha(path)==record['model_sha256'];d=od_records(items);X,names=build_od_features(d,record['config']);assert names==record['features']
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest' if record['config'].get('inference_precision')=='float32' else 'high');ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['features']==names
    if record['architecture'] in ('tabm_route_mixture','tabm_joint_route_time'):
        from modelo_mezcla_rutas import make_duration_model
        m=make_duration_model(ck,record['config']).cuda()
    else:m=CityTabM(ck['mean'],ck['std'],ck['keep'],ck['bins'],ck['center'],record['config']['k'],record['config']['width']).cuda()
    m.load_state_dict(ck['state']);m.eval();out=[]
    with torch.no_grad():
        for first in range(0,len(X),2048):
            block=torch.as_tensor(X[first:first+2048],device='cuda');real_size=len(block)
            if record['config'].get('canonical_batch')==2048 and real_size<2048:block=torch.cat([block,block[-1:].expand(2048-real_size,-1)],0)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=record['config'].get('inference_precision')!='float32'):pred=m(block)
            values=m.quantiles(pred[0],(.5,),mode=record['config'].get('median_mode','discrete'))[:,0] if record['architecture'] in ('tabm_route_mixture','tabm_joint_route_time') else pred.float().mean(1)
            out.append(values[:real_size].cpu().numpy())
    return np.maximum(np.concatenate(out),1),X,names

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    source=Path(a.input).resolve();output=Path(a.output).resolve();assert source.is_relative_to(ROOT) and output.is_relative_to(ROOT);assert not output.exists(),'Output already exists'
    items=json.loads(source.read_text(encoding='utf-8-sig'));pred,_,_=predict_new_od(RUNS/a.id,items)
    write_json(output,dict(model=a.id,model_sha256=sha(RUNS/a.id/'modelo.pt'),input_sha256=sha(source),predictions_seconds=pred,query_observed_path_used=False,code_hashes={name:sha(BASE/name) for name in ('consulta_od.py','consulta_mapa2014.py','predecir_od.py','modelo_tabm.py','modelo_mezcla_rutas.py','features.py')}));print(pred.tolist())

