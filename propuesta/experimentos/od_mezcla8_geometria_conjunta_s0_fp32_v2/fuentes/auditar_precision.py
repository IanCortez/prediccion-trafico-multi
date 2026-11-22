from common import *
from features import *
from inferencia import *
import argparse,shutil,gc,torch
p=argparse.ArgumentParser();p.add_argument('--ids',nargs='+',required=True);a=p.parse_args()
for source in a.ids:
    folder=RUNS/source;original=json.loads((folder/'resultado.json').read_text(encoding='utf-8-sig'));assert original['architecture'] in ('tabm','tabm_route_mixture','tabm_joint_route_time','trajectory_transformer')
    name=source+'_fp32_v2';run=RUNS/name;run.mkdir(exist_ok=False);cfg=original['config'].copy();cfg.update(id=name,inference_precision='float32',trained_model_source=source)
    if original['architecture'] in ('tabm_route_mixture','tabm_joint_route_time'):cfg['canonical_batch']=2048
    snapshot(run,cfg);started=time.time()
    q=None
    if original['architecture'] in ('tabm_route_mixture','tabm_joint_route_time'):
        cfg['canonical_batch']=2048;d,q,_=predict_mixture_quantiles(folder,'validation');pred=q[:,1]
    else:d,pred=predict_run(folder,'validation',precision_override='float32')
    z=np.load(folder/'validacion.npz');assert np.array_equal(d['row'],z['row'])
    difference=np.abs(pred-z['pred']);shutil.copy2(folder/'modelo.pt',run/'modelo.pt')
    np.savez_compressed(run/'validacion.npz',row=d['row'],timestamp=d['timestamp'],pred=pred,true=z['true'])
    rec={**original,'id':name,'config':cfg,'kind':'inference precision audit; no retraining','trained_model_source':source,'validation':metrics(z['true'],pred),'interval':block_interval(z['true'],pred,d['timestamp']),'elapsed_seconds':original['elapsed_seconds']+time.time()-started,'precision_audit':dict(max_change_s=float(difference.max()),mean_change_s=float(difference.mean()),mae_change_s=float(metrics(z['true'],pred)['mae_s']-original['validation']['mae_s'])),'conclusion':'Mismo checkpoint elegido con validación bfloat16; reevaluación fija float32 para reducir dependencia del tamaño de lote. Se conserva la corrida anterior, sin reentrenar ni seleccionar la precisión por su MAE.'}
    if q is not None:
        rec['coverage80']=float(np.mean((z['true']>=q[:,0])&(z['true']<=q[:,2])));rec['width80_s']=float(np.mean(q[:,2]-q[:,0]));np.savez_compressed(run/'validacion.npz',row=d['row'],timestamp=d['timestamp'],pred=pred,true=z['true'],quantiles=q)
    write_json(run/'resultado.json',rec);register(rec);print(dict(id=name,validation=rec['validation'],precision=rec['precision_audit']),flush=True);gc.collect();torch.cuda.empty_cache()


