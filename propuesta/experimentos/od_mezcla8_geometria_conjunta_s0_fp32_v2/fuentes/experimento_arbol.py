from common import *
from features import *
import argparse
from catboost import CatBoostRegressor,Pool
p=argparse.ArgumentParser();p.add_argument("--id",required=True);p.add_argument("--depth",type=int,default=8);p.add_argument("--iterations",type=int,default=3500);p.add_argument("--lr",type=float,default=.05);p.add_argument("--seed",type=int,default=0);p.add_argument("--enriched",action="store_true");p.add_argument("--residual",action="store_true");p.add_argument("--loss",choices=["MAE","RMSE"],default="MAE");p.add_argument("--train-n",type=int,default=0)
p.add_argument("--target-scale",type=float,default=1.0)
p.add_argument("--routes",action="store_true")
p.add_argument("--context",action="store_true")
p.add_argument("--history",action="store_true")
p.add_argument("--decay-days",type=float,default=0)
p.add_argument("--gps-routes",action="store_true")
a=p.parse_args();run=RUNS/a.id;run.mkdir(exist_ok=False);started=time.time();snapshot(run,vars(a))
tr,y,va,v=load_development();X,names=features(tr,a.enriched);V,_=features(va,a.enriched)
if a.routes:
    rr=np.load(DATA/"train_rutas.npz");vv=np.load(DATA/"validation_rutas.npz")
    assert np.array_equal(rr["row"],tr["row"]) and np.array_equal(vv["row"],va["row"])
    X=np.column_stack([X,rr["X"]]);V=np.column_stack([V,vv["X"]]);names+=rr["names"].tolist()
if a.context:
    rr=np.load(DATA/"train_contexto.npz");vv=np.load(DATA/"validation_contexto.npz")
    assert np.array_equal(rr["row"],tr["row"]) and np.array_equal(vv["row"],va["row"])
    X=np.column_stack([X,rr["X"]]);V=np.column_stack([V,vv["X"]]);names+=rr["names"].tolist()
if a.history:
    rr=np.load(DATA/"train_historial.npz");vv=np.load(DATA/"validation_historial.npz")
    assert np.array_equal(rr["row"],tr["row"]) and np.array_equal(vv["row"],va["row"])
    X=np.column_stack([X,rr["X"]]);V=np.column_stack([V,vv["X"]]);names+=rr["names"].tolist()
if a.gps_routes:
    rr=np.load(DATA/"train_gps_rutas.npz");vv=np.load(DATA/"validation_gps_rutas.npz")
    assert np.array_equal(rr["row"],tr["row"]) and np.array_equal(vv["row"],va["row"])
    X=np.column_stack([X,rr["X"]]);V=np.column_stack([V,vv["X"]]);names+=rr["names"].tolist()
ix=np.arange(len(y))
if a.train_n:ix=np.sort(np.random.default_rng(a.seed).choice(ix,min(a.train_n,len(ix)),replace=False))
weights=np.ones(len(y),dtype=np.float64)
if a.decay_days:
    weights=np.exp(-(tr["timestamp"].max()-tr["timestamp"])/(a.decay_days*86400));weights/=weights.mean()
coef=fit_power({k:z[ix] for k,z in tr.items()},y[ix]);base=power_predict(tr,coef);vb=power_predict(va,coef)
for label,pred in (("median",np.full(len(v),np.median(y[ix]))),("power",vb)):
    print(label,metrics(v,pred),flush=True)
target=y-base if a.residual else y
vt=v-vb if a.residual else v
model=CatBoostRegressor(loss_function=a.loss,eval_metric="MAE",iterations=a.iterations,learning_rate=a.lr,depth=a.depth,l2_leaf_reg=5,random_seed=a.seed,task_type="GPU",devices="0",thread_count=4,gpu_ram_part=.75,train_dir=str(run/"catboost"),od_type="Iter",od_wait=250,allow_writing_files=True)
model.fit(Pool(X[ix],target[ix]/a.target_scale,weight=weights[ix],feature_names=names),eval_set=Pool(V,vt/a.target_scale,feature_names=names),use_best_model=True,verbose=200)
pred=model.predict(V)*a.target_scale+(vb if a.residual else 0);pred=np.maximum(pred,1)
model.save_model(str(run/"modelo.cbm"))
np.savez_compressed(run/"validacion.npz",row=va["row"],pred=pred,true=v,timestamp=va["timestamp"])
rec=dict(id=a.id,architecture="OD CatBoost",config=vars(a),power_coefficients=coef,validation=metrics(v,pred),interval=block_interval(v,pred,va["timestamp"]),diagnostic=diagnose(va,v,pred),train_n=len(ix),effective_weighted_n=float(weights[ix].sum()**2/(weights[ix]**2).sum()),validation_n=len(v),features=names,best_iteration=model.get_best_iteration(),elapsed_seconds=time.time()-started,model_sha256=sha(run/"modelo.cbm"),data_manifest_sha256=sha(DATA/"manifiesto.json"),code_sha256=sha(__file__))
rec["conclusion"]="Referencia OD entrenada y seleccionada solo por validación. La cifra de prueba permanece sin consultar. Revisar sesgo por distancia antes de añadir complejidad."
write_json(run/"resultado.json",rec);register(rec)
write_json(run/"importancia.json",dict(sorted(zip(names,model.feature_importances_.tolist()),key=lambda z:-z[1])))
print(json.dumps(rec,ensure_ascii=False),flush=True)







