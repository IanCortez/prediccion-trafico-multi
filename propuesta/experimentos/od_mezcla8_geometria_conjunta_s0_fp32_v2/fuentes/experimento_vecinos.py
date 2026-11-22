from common import *
from features import *
from scipy.spatial import cKDTree
import torch
started=time.time();run=RUNS/"od_vecinos_historicos_s0";run.mkdir(exist_ok=False);snapshot(run,{})
tr,y,va,v=load_development();X,names=features(tr);V,_=features(va)
tree=cKDTree(X[:,:4],leafsize=32)
# Queries use only OD; travel times are fetched exclusively from training.
distance,ix=tree.query(V[:,:4],k=512,workers=4)
np.savez_compressed(run/"banco.npz",coordinates=X[:,:4],duration=y,od_m=tr["od_m"],timestamp=tr["timestamp"],hour=X[:,14],dow=X[:,15])
device="cuda";torch.set_num_threads(4)
idx=torch.as_tensor(ix.astype(np.int32),device=device,dtype=torch.long);del ix
dist=torch.as_tensor(distance,device=device,dtype=torch.float32);del distance
yt=torch.as_tensor(y,device=device)[idx]
trainod=torch.as_tensor(tr["od_m"],device=device)[idx];queryod=torch.as_tensor(va["od_m"],device=device)[:,None]
trainhour=torch.as_tensor(X[:,14],device=device)[idx];qh=torch.as_tensor(V[:,14],device=device)[:,None]
dh=(trainhour-qh).abs();dh=torch.minimum(dh,24-dh)
td=torch.as_tensor(X[:,15],device=device)[idx];qd=torch.as_tensor(V[:,15],device=device)[:,None]
same_type=(td>=5)==(qd>=5)
age=torch.as_tensor(va["timestamp"],device=device,dtype=torch.float32)[:,None]-torch.as_tensor(tr["timestamp"],device=device,dtype=torch.float32)[idx]
# Candidate family tests local route reuse, distance calibration, and time matching.
configs=[dict(k=k,exponent=e,hour_bw=h,spatial_bw=b,age_days=90) for k,e,h,b in [
(32,0,4,.25),(128,0,4,.25),(512,0,4,.25),(128,.46,4,.25),(512,.46,4,.25),(128,1,4,.25),(512,.46,2,.25),(512,.46,6,.25),(512,.46,4,.5),(512,.46,4,.15)]]
best=float("inf");runs=[]
for ci,cfg in enumerate(configs):
    k=cfg["k"];scale=(queryod/trainod[:,:k]).clamp(.25,4)**cfg["exponent"]
    candidate=yt[:,:k]*scale
    w=torch.exp(-.5*(dist[:,:k]/cfg["spatial_bw"])**2-.5*(dh[:,:k]/cfg["hour_bw"])**2-age[:,:k]/(cfg["age_days"]*86400))
    w=w*torch.where(same_type[:,:k],1.,.35)+1e-20
    values,order=candidate.sort(1);sw=w.gather(1,order);cs=sw.cumsum(1)
    mid=(cs<cs[:,-1:]/2).sum(1).clamp_max(k-1)
    pred=values.gather(1,mid[:,None]).squeeze(1).cpu().numpy()
    vm=metrics(v,pred);rec=dict(config=cfg,validation=vm);runs.append(rec);print(json.dumps(rec),flush=True)
    if vm["mae_s"]<best:
        best=vm["mae_s"];chosen=cfg
        np.savez_compressed(run/"validacion.npz",row=va["row"],pred=pred,true=v,timestamp=va["timestamp"])
write_json(run/"candidatos.json",runs);z=np.load(run/"validacion.npz")
rec=dict(id=run.name,architecture="OD historical weighted median neighbors",config=chosen,validation=metrics(v,z["pred"]),interval=block_interval(v,z["pred"],va["timestamp"]),diagnostic=diagnose(va,v,z["pred"]),train_n=len(y),elapsed_seconds=time.time()-started,conclusion="Comprueba reutilización de rutas/tiempos históricos con OD próximo y hora comparable; no utiliza etiquetas de consulta. Comparar con árboles antes de aceptar un híbrido.")
write_json(run/"resultado.json",rec);register(rec);print(json.dumps(rec),flush=True)

