"""Recalculate saved final Train-pilot metrics without training, GPU inference or Test access."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DEV=ROOT/"propuesta/replanteamiento_20261121"
def read(path):return json.loads(path.read_text(encoding="utf-8-sig"))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    closure=read(DEV/"CIERRE_ULTIMA_PROPUESTA.json")
    assert closure["status"]=="COMPLETE_AND_CLOSED_BY_USER"
    assert closure["target_achieved"] is False and closure["Test"]=="SEALED"
    assert sha(DEV/"RESUMEN_FINAL_ULTIMA_PROPUESTA.txt")==closure["summary_sha256"]
    run=DEV/"red_cinetica_Train_s0"
    record=read(run/"resultado.json")
    assert record["complete"] and record["Validation"] is False and record["Test"] is False
    with np.load(run/"predicciones_held_Train.npz",allow_pickle=False) as data:
        saved={k:data[k] for k in data.files}
    assert len(saved["row"])==len(np.unique(saved["row"]))==10000
    assert sha(run/"predicciones_held_Train.npz")==record["predictions_sha256"]
    computed={}
    for arm in ("position","kinetic","reference"):
        pred=saved[arm].astype(float);true=saved["true"].astype(float)
        assert pred.shape==true.shape==(10000,) and np.isfinite(pred).all()
        value=float(np.abs(pred-true).mean())
        assert abs(value-record["metrics"][arm]["mae_s"])<1e-10
        computed[arm]=value
        if arm=="reference":continue
        with np.load(run/arm/"predicciones_held_Train.npz",allow_pickle=False) as data:
            assert np.array_equal(data["row"],saved["row"])
            assert np.array_equal(data["true"],saved["true"])
            assert np.array_equal(data["pred"],saved[arm])
            assert np.array_equal(data["quantiles"][:,1],saved[arm])
            prob=np.exp(data["log_prob"])
            assert np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=1e-10,rtol=0)
            assert int((~data["supported"]).sum())==76
            assert np.array_equal(data["pred"][~data["supported"]],saved["reference"][~data["supported"]])
        weights=record["weights"][arm]
        assert sha(run/arm/"epoca_80.pt")==weights["sha256"]
    ref=read(DEV/"referencia_publica_Train_red_vial.json")
    assert sha(DEV/ref["id"]/"epoca_4.pt")==ref["model_sha256"]
    proof=read(run/"reproduccion_independiente.json")
    assert proof["passed"] and proof["reference_all10000_exact"]
    for arm in ("position","kinetic"):
        assert proof["arms"][arm]["passed"] and proof["arms"][arm]["all10000_quantiles_exact"]
    assert computed["kinetic"]>computed["reference"]
    assert record["hypothesis_position_passed"] is False and record["hypothesis_kinetic_passed"] is False
    print(json.dumps({"passed":True,"closed":True,"scope":"10000 reused held-Train queries, not primary Validation",
                      "MAE_s":computed,"best_primary_Validation_MAE_s":129.99003446901307,
                      "target_achieved":False,"Test_read":False,"GPU_used":False,"training_run":False},indent=2))
if __name__=="__main__":main()
