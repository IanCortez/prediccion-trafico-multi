from common import *
from features import *
from inferencia import feature_input
import argparse
p=argparse.ArgumentParser();p.add_argument("--id",required=True);p.add_argument("--engine",choices=["xgboost","lightgbm"],required=True);p.add_argument("--seed",type=int,default=0);p.add_argument("--iterations",type=int,default=6000)
p.add_argument("--gps-routes",action="store_true")
p.add_argument("--map-geometry",action="store_true")
p.add_argument("--weekly-only",action="store_true")
p.add_argument("--dynamic-routes",action="store_true")
p.add_argument("--hourly-gps",action="store_true")
p.add_argument("--archive-maps",nargs="+",choices=["2014","2026"],default=[])
p.add_argument("--calendar",action="store_true")
p.add_argument("--projected-routes",action="store_true");p.add_argument("--mapmatched",action="store_true")
p.add_argument("--learned-route",action="store_true")
p.add_argument("--road-attributes",action="store_true")
a=p.parse_args();run=RUNS/a.id;run.mkdir(exist_ok=False);snapshot(run,vars(a));started=time.time()
tr,X,names=feature_input("train",dict(config=vars(a),architecture=a.engine));va,V,_=feature_input("validation",dict(config=vars(a),architecture=a.engine));y=np.load(DATA/"train_y.npy");v=np.load(DATA/"validation_y.npy")
if a.engine=="xgboost":
    import xgboost as xgb
    model=xgb.XGBRegressor(objective="reg:absoluteerror",eval_metric="mae",n_estimators=a.iterations,learning_rate=.04,max_depth=8,min_child_weight=50,reg_lambda=20,subsample=.8,colsample_bytree=.9,tree_method="hist",device="cuda",n_jobs=4,early_stopping_rounds=200,random_state=a.seed)
    model.fit(X,y,eval_set=[(V,v)],verbose=200);file="modelo.ubj";model.save_model(run/file);selected=model.best_iteration
else:
    import lightgbm as lgb
    model=lgb.LGBMRegressor(objective="regression_l1",n_estimators=a.iterations,learning_rate=.03,num_leaves=63,min_child_samples=50,reg_lambda=10,colsample_bytree=.9,subsample=.8,subsample_freq=1,n_jobs=4,random_state=a.seed,verbosity=-1)
    model.fit(X,y,eval_set=[(V,v)],eval_metric="l1",callbacks=[lgb.early_stopping(200,verbose=False),lgb.log_evaluation(200)])
    file="modelo.txt";model.booster_.save_model(str(run/file));selected=model.best_iteration_
pred=np.maximum(model.predict(V),1)
np.savez_compressed(run/"validacion.npz",row=va["row"],pred=pred,true=v,timestamp=va["timestamp"])
rec=dict(id=a.id,architecture=a.engine,config=vars(a),validation=metrics(v,pred),interval=block_interval(v,pred,va["timestamp"]),diagnostic=diagnose(va,v,pred),train_n=len(y),features=names,selected_iteration=selected,elapsed_seconds=time.time()-started,model_file=file,model_sha256=sha(run/file),data_manifest_sha256=sha(DATA/"manifiesto.json"),conclusion="Comparador con las mismas entradas OD, mapa y contexto. Aísla la restricción de particiones simétricas de CatBoost frente a árboles de otra estructura.")
write_json(run/"resultado.json",rec);register(rec);print(json.dumps(rec),flush=True)









