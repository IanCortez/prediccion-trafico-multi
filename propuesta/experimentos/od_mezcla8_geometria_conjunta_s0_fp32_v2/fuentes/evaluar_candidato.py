from common import *
from features import *
from inferencia import *
candidate=json.loads((BASE/"CANDIDATO.json").read_text(encoding="utf-8-sig"))
assert candidate["manifest_sha256"]==sha(DATA/"manifiesto.json")
assert not (BASE/"CONFIRMACION.json").exists(),"Prueba ya expuesta; no repetir selección sobre este conjunto"
for name,value in candidate["inference_code_hashes"].items():assert sha(BASE/name)==value,"Inference code changed after freeze"
for path,value in candidate["test_input_hashes"].items():assert sha(ROOT/path)==value,"Test features changed after freeze"
assert not (BASE/"instrumentos/PRUEBA_ABIERTA.json").exists(),"La prueba ya fue abierta; no repetir aunque la ejecución anterior se interrumpiera"
preds=[];d=None
for rec in candidate["models"]:
    folder=RUNS/rec["id"];assert sha(folder/rec["model_file"])==rec["sha256"];assert sha(folder/"resultado.json")==rec["result_sha256"]
    data,pred=predict_run(folder,"test")
    if d is not None:assert np.array_equal(data["row"],d["row"])
    d=data;preds.append(pred)
pred=(np.mean if candidate["aggregation"]=="mean" else np.median)(np.stack(preds),axis=0)
assert np.isfinite(pred).all()
np.savez_compressed(BASE/"predicciones_confirmacion_sin_etiquetas.npz",row=d["row"],pred=pred)
with (BASE/"instrumentos/accesos_prueba.jsonl").open("a",encoding="utf-8") as f:
    f.write(json.dumps(dict(time=time.time(),candidate_sha256=sha(BASE/"CANDIDATO.json"),prediction_sha256=sha(BASE/"predicciones_confirmacion_sin_etiquetas.npz")))+"\n")
# First read of held-out numerical labels, after committing predictions and configuration.
with (BASE/"instrumentos/PRUEBA_ABIERTA.json").open("x",encoding="utf-8") as f:
    json.dump(dict(time=time.time(),candidate_sha256=sha(BASE/"CANDIDATO.json")),f)
manifest=json.loads((DATA/"manifiesto.json").read_text())
assert sha(DATA/"test_y.npy")==manifest["test"]["y_sha256"],"Test-label integrity changed"
y=np.load(DATA/"test_y.npy");mask=~d["exposed"]
assert np.all(y%15==0) and y.min()>=60 and y.max()<=1785
m=metrics(y[mask],pred[mask]);interval=block_interval(y[mask],pred[mask],d["timestamp"][mask])
out=dict(candidate_sha256=sha(BASE/"CANDIDATO.json"),confirmation=m,interval=interval,diagnostic=diagnose({k:z[mask] for k,z in d.items()},y[mask],pred[mask]),excluded_previously_exposed=int((~mask).sum()),point_goal_achieved=m["mae_s"]<73.5,strong_goal_achieved=interval["ci95_s"][1]<73.5,comparison="Operational threshold on explicit local cohort; not exact reproduction of TransferTraj")
write_json(BASE/"CONFIRMACION.json",out);print(json.dumps(out,indent=2),flush=True)


