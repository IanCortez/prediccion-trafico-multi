from common import *
from features import *
def feature_input(split,record):
    cfg=record["config"];d=dict(np.load(DATA/(split+"_x.npz")));X,names=features(d,cfg.get("enriched",False),calendar=cfg.get("calendar",False))
    arch=record["architecture"]
    if arch in ("xgboost","lightgbm"):
        components=["rutas","contexto"]+(["gps_rutas"] if cfg.get("gps_routes",False) else [])+(["geometria_rutas"] if cfg.get("map_geometry",False) else [])+(["rutas_dinamicas"] if cfg.get("dynamic_routes",False) else [])+(["gps_horario"] if cfg.get("hourly_gps",False) else [])
    else:
        components=[name for flag,name in (("routes","rutas"),("context","contexto"),("history","historial"),("gps_routes","gps_rutas"),("map_geometry","geometria_rutas"),("dynamic_routes","rutas_dinamicas"),("hourly_gps","gps_horario")) if cfg.get(flag,False)]
    components += ["mapa_"+str(year) for year in cfg.get("archive_maps",[])]
    if cfg.get("projected_routes",False) or cfg.get("mapmatched",False):components.append("rutas_proyectadas2014")
    if cfg.get("mapmatched",False):components.append("gps_mapmatched")
    if cfg.get("learned_route",False):components.append("ruta_preferida")
    if cfg.get("road_attributes",False):components.append("atributos_rutas2014")
    for component in components:
        z=np.load(DATA/f"{split}_{component}.npz");assert np.array_equal(d["row"],z["row"])
        X=np.column_stack([X,z["X"]]);names+=z["names"].tolist()
    if cfg.get("weekly_only",False):
        keep=[i for i,n in enumerate(names) if n not in ("month","season_sin","season_cos")]
        X=X[:,keep];names=[names[i] for i in keep]
    return d,X,names
def model_path(run,record):
    if record.get("model_file"):return Path(run)/record["model_file"]
    if (Path(run)/"modelo.cbm").exists():return Path(run)/"modelo.cbm"
    return Path(run)/"modelo.pt"
def predict_run(run,split="validation",precision_override=None):
    run=Path(run);r=json.loads((run/"resultado.json").read_text(encoding="utf-8-sig"))
    assert r.get("cohort","primary")=="primary","Privileged/diagnostic cohort is ineligible for OD confirmation"
    cfg=r["config"].copy();path=model_path(run,r)
    if precision_override is not None:cfg["inference_precision"]=precision_override
    assert r.get("model_sha256",sha(path))==sha(path)
    if r["architecture"] in ("trajectory_transformer","trajectory_route_aux","duration_distribution","dense_suffix"):
        import torch
        from modelo_trayectorias import TrajectoryEncoder,RouteAuxEncoder,DurationClassifier,TaskAwareEncoder
        torch.set_num_threads(4);torch.set_float32_matmul_precision("highest" if cfg.get("inference_precision")=="float32" else "high")
        d=dict(np.load(DATA/(split+"_x.npz")));f,_=features(d);ck=torch.load(path,map_location="cpu",weights_only=False)
        grid=ck["state"]["context_grid"]
        cls=RouteAuxEncoder if r["architecture"]=="trajectory_route_aux" else DurationClassifier if r["architecture"]=="duration_distribution" else TaskAwareEncoder if r["architecture"]=="dense_suffix" and cfg.get("task_adapter",False) else TrajectoryEncoder
        model=cls(grid,ck["grid_info"],ck["coef"]).cuda()
        model.load_state_dict(ck["state"]);model.eval();out=[]
        xy=np.stack([f[:,:2],f[:,2:4]],axis=1);dep=f[:,18:24]
        with torch.no_grad():
            for start in range(0,len(xy),8192):
                x=torch.as_tensor(xy[start:start+8192],device="cuda");t=torch.as_tensor(dep[start:start+8192],device="cuda")
                known=torch.zeros(x.shape[:2],device="cuda");visible=torch.zeros_like(known,dtype=torch.bool);visible[:,0]=True
                with torch.autocast("cuda",dtype=torch.bfloat16,enabled=cfg.get("inference_precision")!="float32"):
                    if r["architecture"]=="duration_distribution":p=model.quantiles(model.posterior(x,t),(.5,))
                    else:p,_=model(x,known,visible,t)
                out.append((p[:,0] if r["architecture"]=="duration_distribution" else p[:,1]).cpu().numpy())
        pred=np.concatenate(out)
    else:
        d,X,names=feature_input(split,r)
        assert names==r["features"],"Feature schema/order mismatch"
        if path.suffix==".cbm":
            from catboost import CatBoostRegressor
            m=CatBoostRegressor();m.load_model(str(path));assert m.feature_names_==names
            pred=m.predict(X)*cfg.get("target_scale",1)
            if cfg.get("residual",False):pred+=power_predict(d,r["power_coefficients"])
        elif r["architecture"]=="xgboost":
            import xgboost as xgb
            m=xgb.Booster();m.load_model(path)
            pred=m.predict(xgb.DMatrix(X),iteration_range=(0,r["selected_iteration"]+1))
        elif r["architecture"]=="lightgbm":
            import lightgbm as lgb
            m=lgb.Booster(model_file=str(path));pred=m.predict(X,num_iteration=r["selected_iteration"],num_threads=4)
        elif r["architecture"] in ("tabm","tabm_route_mixture","tabm_joint_route_time"):
            import torch
            from modelo_tabm import CityTabM
            from modelo_mezcla_rutas import RouteMixtureTabM,make_duration_model
            torch.set_num_threads(4);torch.set_float32_matmul_precision("highest" if cfg.get("inference_precision")=="float32" else "high")
            ck=torch.load(path,map_location="cpu",weights_only=False)
            if r["architecture"] in ("tabm_route_mixture","tabm_joint_route_time"):m=make_duration_model(ck,cfg).cuda()
            else:m=CityTabM(ck["mean"],ck["std"],ck["keep"],ck["bins"],ck["center"],cfg["k"],cfg["width"]).cuda()
            m.load_state_dict(ck["state"]);m.eval()
            out=[]
            with torch.no_grad():
                for first in range(0,len(X),2048):
                    block=torch.as_tensor(X[first:first+2048],device="cuda");real_size=len(block)
                    if cfg.get("canonical_batch")==2048 and real_size<2048:block=torch.cat([block,block[-1:].expand(2048-real_size,-1)],0)
                    with torch.autocast("cuda",dtype=torch.bfloat16,enabled=cfg.get("inference_precision")!="float32"):values=m(block)
                    out.append((m.quantiles(values[0],(.5,),mode=cfg.get('median_mode','discrete'))[:,0] if r["architecture"] in ("tabm_route_mixture","tabm_joint_route_time") else values.float().mean(1))[:real_size].cpu().numpy())
            pred=np.concatenate(out)
        elif r["architecture"]=="OD physical residual multiscale":
            import torch
            from modelo_residual import build_model
            torch.set_num_threads(4);torch.set_float32_matmul_precision("highest")
            ck=torch.load(path,map_location="cpu",weights_only=False)
            assert names==ck["features"]
            m=build_model(ck["mean"],ck["std"],ck["coef"],cfg).cuda()
            m.load_state_dict(ck["state"]);m.eval()
            with torch.no_grad():pred=np.concatenate([m(torch.as_tensor(z,device="cuda")).cpu().numpy() for z in np.array_split(X,max(1,int(np.ceil(len(X)/8192))))])
        else:raise ValueError("This baseline has no certified inference adapter: "+r["architecture"])
    pred=np.maximum(pred,1);assert np.isfinite(pred).all()
    return d,pred


def prediction_input_paths(record,split):
    cfg=record["config"];paths=[DATA/(split+"_x.npz")]
    if record["architecture"] in ("trajectory_transformer","trajectory_route_aux","duration_distribution","dense_suffix"):return paths
    if record["architecture"] in ("xgboost","lightgbm"):
        components=["rutas","contexto"]+(["gps_rutas"] if cfg.get("gps_routes",False) else [])+(["geometria_rutas"] if cfg.get("map_geometry",False) else [])+(["rutas_dinamicas"] if cfg.get("dynamic_routes",False) else [])+(["gps_horario"] if cfg.get("hourly_gps",False) else [])
    else:components=[name for flag,name in (("routes","rutas"),("context","contexto"),("history","historial"),("gps_routes","gps_rutas"),("map_geometry","geometria_rutas"),("dynamic_routes","rutas_dinamicas"),("hourly_gps","gps_horario")) if cfg.get(flag,False)]
    components += ["mapa_"+str(year) for year in cfg.get("archive_maps",[])]
    if cfg.get("projected_routes",False) or cfg.get("mapmatched",False):components.append("rutas_proyectadas2014")
    if cfg.get("mapmatched",False):components.append("gps_mapmatched")
    if cfg.get("learned_route",False):components.append("ruta_preferida")
    if cfg.get("road_attributes",False):components.append("atributos_rutas2014")
    return paths+[DATA/f"{split}_{name}.npz" for name in components]














def predict_mixture_quantiles(run,split='validation',levels=(.1,.5,.9),precision_override='float32',canonical_batch=2048,include_distribution=False):
    import torch
    from modelo_mezcla_rutas import RouteMixtureTabM,make_duration_model
    run=Path(run);r=json.loads((run/'resultado.json').read_text(encoding='utf-8-sig'));assert r['architecture'] in ('tabm_route_mixture','tabm_joint_route_time');cfg=r['config'].copy();cfg['inference_precision']=precision_override
    path=model_path(run,r);assert sha(path)==r['model_sha256'];d,X,names=feature_input(split,r);assert names==r['features'];torch.set_num_threads(4);torch.set_float32_matmul_precision('highest' if precision_override=='float32' else 'high')
    ck=torch.load(path,map_location='cpu',weights_only=False);m=make_duration_model(ck,cfg).cuda();m.load_state_dict(ck['state']);m.eval();quantiles=[];weights=[];distributions=[]
    with torch.no_grad():
        for first in range(0,len(X),canonical_batch):
            block=torch.as_tensor(X[first:first+canonical_batch],device='cuda');size=len(block)
            if size<canonical_batch:block=torch.cat([block,block[-1:].expand(canonical_batch-size,-1)],0)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision_override!='float32'):output=m(block)
            logp,w=output[:2]
            quantiles.append(m.quantiles(logp,levels,mode=cfg.get('median_mode','discrete'))[:size].cpu().numpy());weights.append(w.exp().mean(1)[:size].cpu().numpy())
            if include_distribution:distributions.append(logp.exp().mean(1)[:size].cpu().numpy())
    result=(d,np.concatenate(quantiles),np.concatenate(weights))
    return (*result,np.concatenate(distributions)) if include_distribution else result

