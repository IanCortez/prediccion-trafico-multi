from common import *
from features import *
from catboost import CatBoostRegressor,Pool
run=RUNS/"ORACULO_ruta_observada9_s0";run.mkdir(exist_ok=False);snapshot(run,{})
started=time.time();tr,y,va,v=load_development();X,names=features(tr);V,_=features(va)
a=np.load(DATA/"geometria_train.npz");b=np.load(DATA/"geometria_validation.npz")
pos=np.searchsorted(a["row"],tr["row"]);safe=np.minimum(pos,len(a["row"])-1);valid=(pos<len(a["row"]))&(a["row"][safe]==tr["row"])
ix=np.flatnonzero(valid)
def privileged(f,geometry):
    q=np.linspace(0,1,9);o=f[:,:2];d=f[:,2:4]
    pts=o[:,None,:]*(1-q[None,:,None])+d[:,None,:]*q[None,:,None]
    pts[:,1:-1,:]+=geometry[:,1:15].reshape(-1,7,2)
    coarse=np.sqrt(np.sum(np.diff(pts,axis=1)**2,axis=2)).sum(1)
    return np.column_stack([coarse,np.log(np.maximum(coarse/(f[:,6]),1)),geometry[:,1:15]])
P=privileged(X[ix],a["X"][pos[ix]]);Q=privileged(V,b["X"])
X=np.column_stack([X[ix],P]);V=np.column_stack([V,Q]);y=y[ix]
# Fixed 9-point arclength geometry removes variable point count and sampling timing.
names+=["observed_9point_length_km","observed_9point_circuity_log"]+[f"observed_shape_{i}" for i in range(14)]
model=CatBoostRegressor(loss_function="MAE",eval_metric="MAE",iterations=4500,learning_rate=.05,depth=10,l2_leaf_reg=5,random_seed=0,task_type="GPU",devices="0",thread_count=4,gpu_ram_part=.3,train_dir=str(run/"catboost"),od_type="Iter",od_wait=250)
model.fit(Pool(X,y/600,feature_names=names),eval_set=Pool(V,v/600,feature_names=names),use_best_model=True,verbose=200)
pred=np.maximum(model.predict(V)*600,1);model.save_model(str(run/"modelo.cbm"))
np.savez_compressed(run/"validacion.npz",row=va["row"],pred=pred,true=v,timestamp=va["timestamp"])
rec=dict(id=run.name,cohort="privileged observed-route diagnostic",architecture="CatBoost oracle, not OD",config=dict(observed_route=True,point_count=False,arrival_time=False),validation=metrics(v,pred),interval=block_interval(v,pred,va["timestamp"]),train_n=len(y),features=names,elapsed_seconds=time.time()-started,conclusion="Control privilegiado de ruta GPS observada, remuestreada a exactamente 9 puntos por arclongitud. No es una solución OD ni una cota teórica; no puede usarse para declarar la meta.")
write_json(run/"resultado.json",rec);register(rec);print(json.dumps(rec),flush=True)
# Diagnose where the strict OD model loses information; no redefinition of primary cohort.
r=RUNS/"od_xgboost_contexto_rutas_s0/validacion.npz";z=np.load(r);error=np.abs(z["pred"]-v)
circ=b["X"][:,0]/va["od_m"];speed=b["X"][:,0]/v*3.6
masks={"all":np.ones(len(v),bool),"circuity_le_2_5":circ<=2.5,"circuity_gt_2_5":circ>2.5,"route_speed_ge_4":speed>=4,"route_speed_lt_4":speed<4,"previous_thesis_quality":(circ<=2.5)&(speed>=4),"gps_jump_gt_500m":b["X"][:,-1]>500}
slices={name:dict(n=int(m.sum()),fraction=float(m.mean()),mae_s=float(error[m].mean()) if m.any() else None,error_share=float(error[m].sum()/error.sum())) for name,m in masks.items()}
write_json(BASE/"instrumentos/diagnostico_ruta_privilegiada.json",dict(strict_od_slices=slices,oracle=rec["validation"],caveat="Validation-only diagnostic conditioned on observed route/label. These conditions are unavailable for OD inference; primary cohort unchanged.",test_labels_read=False))

