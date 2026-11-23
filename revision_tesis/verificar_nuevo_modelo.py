"""Read-only CPU inspection of the complete saved checkpoint chain; no training or Test access."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DEV=ROOT/"propuesta/replanteamiento_20261121"
CHAIN=[("historica_fresca_s0",27),("atencion_historia_individual_s0",3),("atencion_subtrayectos_s0",6)]
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
    return h.hexdigest()
def main():
    import torch
    torch.set_num_threads(2)
    checks=[]
    for run_id,epoch in CHAIN:
        run=DEV/run_id
        record=json.loads((run/"resultado.json").read_text(encoding="utf-8-sig"))
        path=run/f"epoca_{epoch}.pt"
        assert digest(path)==record["model_sha256"]
        state=torch.load(path,map_location="cpu",weights_only=False)
        assert state["epoch"]==epoch,(run_id,state.keys())
        assert "state" in state or "branch_state" in state
        for name,expected in record.get("sources",{}).items():
            assert digest(run/"fuentes"/name)==expected,(run_id,name)
        checks.append({"model":run_id,"epoch":epoch,"sha256":record["model_sha256"]})
        del state
    run=DEV/CHAIN[-1][0]
    record=json.loads((run/"resultado.json").read_text(encoding="utf-8-sig"))
    proof=json.loads((run/"reproduccion_independiente.json").read_text(encoding="utf-8-sig"))
    assert proof["passed"] and proof["all127941quantiles_exact"] and proof["raw65_reverse_singleton_exact"]
    with np.load(run/"validacion.npz",allow_pickle=False) as data:
        assert len(data["pred"])==len(data["true"])==127941
        assert len(np.unique(data["row"]))==127941
        mae=float(np.abs(data["pred"].astype(float)-data["true"].astype(float)).mean())
    assert abs(mae-record["validation"]["mae_s"])<1e-9
    assert abs(mae-129.99003446901307)<1e-9
    print(json.dumps({"passed":True,"checkpoints":checks,"n":127941,"MAE_s":mae,
                      "saved_full_reproduction_passed":True,
                      "new_inference_executed":False,"GPU_used":False,"Test_read":False},indent=2))
if __name__=="__main__":main()
