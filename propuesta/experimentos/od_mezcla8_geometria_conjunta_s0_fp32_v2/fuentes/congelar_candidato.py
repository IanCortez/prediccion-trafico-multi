from common import *
from features import *
from inferencia import *
import argparse
p=argparse.ArgumentParser();p.add_argument("--ids",nargs="+",required=True);p.add_argument("--aggregation",choices=["mean","median"],default="mean")
a=p.parse_args();assert not (BASE/"CANDIDATO.json").exists(),"Ya hay un candidato congelado"
manifest=json.loads((DATA/"manifiesto.json").read_text());assert sha(DATA/"validation_y.npy")==manifest["validation"]["y_sha256"]
truth=np.load(DATA/"validation_y.npy");records=[];preds=[];test_inputs={}
for name in a.ids:
    folder=RUNS/name;r=json.loads((folder/"resultado.json").read_text(encoding="utf-8-sig"))
    assert r["train_n"]==manifest["train"]["n"] and not r["config"].get("enriched",False),"Exige cohorte primaria y OD estricto"
    assert r.get("cohort","primary")=="primary","Control privilegiado/diagnóstico excluido"
    if r['architecture'] in ('tabm','tabm_route_mixture','tabm_joint_route_time','trajectory_transformer'):
        assert r['config'].get('inference_precision')=='float32','Exige auditoría float32 estricta antes de congelar un modelo neuronal'
    if r['architecture'] in ('tabm_route_mixture','tabm_joint_route_time'):
        assert r['config'].get('canonical_batch')==2048,'Fijar lote canónico de la mezcla antes de congelar'
    d,pred=predict_run(folder,"validation");z=np.load(folder/"validacion.npz")
    assert np.array_equal(d["row"],z["row"]) and np.max(np.abs(pred-z["pred"]))<.05
    preds.append(pred);file=model_path(folder,r)
    records.append(dict(id=name,model_file=file.name,sha256=sha(file),result_sha256=sha(folder/"resultado.json"),config=r["config"]))
    for path in prediction_input_paths(r,"test"):test_inputs[str(path.relative_to(ROOT))]=sha(path)
pred=(np.mean if a.aggregation=="mean" else np.median)(np.stack(preds),axis=0);vm=metrics(truth,pred)
assert vm["mae_s"]<73.5,f"Validación aún {vm['mae_s']:.3f}s; mantener prueba reservada"
code={p.name:sha(p) for p in BASE.glob("*.py")}
write_json(BASE/"CANDIDATO.json",dict(models=records,aggregation=a.aggregation,validation=vm,manifest_sha256=sha(DATA/"manifiesto.json"),test_input_hashes=test_inputs,inference_code_hashes=code,created_unix=time.time(),test_reads=0))
print(vm)


