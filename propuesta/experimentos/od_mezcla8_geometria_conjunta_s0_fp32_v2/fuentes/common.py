from pathlib import Path
import os, sys, json, hashlib, time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"propuesta"
DATA=BASE/"datos"
RUNS=BASE/"experimentos"
if not Path(sys.executable).resolve().is_relative_to(ROOT):
    raise RuntimeError("Use el entorno conda local del proyecto")
for p in (DATA,RUNS,BASE/"instrumentos"): p.mkdir(parents=True,exist_ok=True)
os.chdir(ROOT)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()
def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x)),encoding="utf-8")
def metrics(y,p):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
    assert y.ndim==p.ndim==1 and len(y)==len(p)>0 and np.isfinite(y).all() and np.isfinite(p).all(),"TTE metrics require one finite destination duration per trip"
    e=np.abs(y-p)
    return dict(mae_s=float(e.mean()),rmse_s=float(np.sqrt(np.mean((y-p)**2))),mdae_s=float(np.median(e)),mape=float(np.mean(e/np.maximum(y,1))*100),n=len(y))
def register(rec):
    with (BASE/"registro.jsonl").open("a",encoding="utf-8") as f: f.write(json.dumps(rec,ensure_ascii=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x))+"\n")
    from evaluar_iteracion import report_iteration
    report_iteration(rec)


def snapshot(run,config):
    import shutil
    folder=Path(run)/"fuentes";folder.mkdir(exist_ok=True)
    hashes={}
    for p in BASE.glob("*.py"):
        shutil.copy2(p,folder/p.name);hashes[p.name]=sha(p)
    components=[name for flag,name in (("routes","rutas"),("context","contexto"),("history","historial"),("gps_routes","gps_rutas"),("map_geometry","geometria_rutas"),("dynamic_routes","rutas_dinamicas"),("hourly_gps","gps_horario")) if config.get(flag,False)]
    if config.get("engine"):components=list(dict.fromkeys(["rutas","contexto"]+components))
    components += ["mapa_"+str(year) for year in config.get("archive_maps",[])]
    if config.get("projected_routes",False) or config.get("mapmatched",False):components.append("rutas_proyectadas2014")
    if config.get("mapmatched",False):components.append("gps_mapmatched")
    if config.get("learned_route",False):components.append("ruta_preferida")
    if config.get("road_attributes",False):components.append("atributos_rutas2014")
    files=[DATA/(split+"_"+component+".npz") for split in ("train","validation") for component in components]
    if config.get('architecture')=='tabm_joint_route_time' or config.get('route_aux',0)>0 or config.get('aux',0)>0:files.append(DATA/'geometria_train.npz')
    if config.get('pause_weight',0)>0:files.append(DATA/'pausas_gps_train.npz')
    input_hashes={str(p.relative_to(ROOT)):sha(p) for p in files if p.exists()}
    write_json(Path(run)/"ejecucion.json",dict(config=config,input_hashes=input_hashes,code_hashes=hashes,python=sys.version,executable=sys.executable,started_unix=time.time(),manifest_sha256=sha(DATA/"manifiesto.json")))









