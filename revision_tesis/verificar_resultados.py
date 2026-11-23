"""Read-only review of saved Validation results; never trains, imports Torch or opens Test."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()
def main():
    manifest=json.loads((ROOT/"revision_tesis/manifiesto_archivos.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path=ROOT/item["path"]
        assert path.is_file(),item["path"]
        assert path.stat().st_size==item["bytes"],item["path"]
        assert sha(path)==item["sha256"],item["path"]
    roots=[ROOT/"propuesta/experimentos",ROOT/"propuesta/comparaciones_transfertraj",ROOT/"propuesta/replanteamiento_20261121"]
    count=0
    for folder in roots:
        for record_path in sorted(folder.glob("*/resultado.json")):
            record=json.loads(record_path.read_text(encoding="utf-8-sig"))
            if "validation" not in record or record["validation"].get("n")!=127941:continue
            if record.get("conclusion","").lower().find("incomplet")>=0:continue
            predictions=record_path.parent/"validacion.npz"
            if not predictions.exists():continue
            # The initial failed run is retained only as evidence, never as a completed model.
            if record_path.parent.name=="od_transfertraj_adaptado_sin_pre_s0":continue
            with np.load(predictions,allow_pickle=False) as data:
                if not {"pred","true"}.issubset(data.files):continue
                pred=np.asarray(data["pred"],dtype=np.float64)
                true=np.asarray(data["true"],dtype=np.float64)
                if pred.ndim==2:
                    assert {"alphas","selected_alpha"}.issubset(data.files),record_path.parent.name
                    selected=float(record["selected_alpha"])
                    assert float(data["selected_alpha"])==selected
                    matches=np.flatnonzero(np.asarray(data["alphas"])==selected)
                    assert len(matches)==1
                    if "all_alpha_validation" in record:
                        all_mae=np.abs(pred-true[None,:]).mean(1)
                        assert np.allclose(all_mae,[r["mae_s"] for r in record["all_alpha_validation"]],atol=1e-5,rtol=0)
                    pred=pred[int(matches[0])]
                assert pred.ndim==true.ndim==1 and len(pred)==len(true)==127941,record_path.parent.name
                assert np.isfinite(pred).all() and np.isfinite(true).all()
                if "row" in data:assert len(np.unique(data["row"]))==len(pred)
                mae=float(np.abs(pred-true).mean())
            expected=float(record["validation"]["mae_s"])
            assert abs(mae-expected)<1e-5,(record_path.parent.name,mae,expected)
            print(f"{record_path.parent.name}: n={len(pred)} MAE={mae:.6f} s")
            count+=1
    assert count>=8,count
    print(f"OK: {len(manifest['files'])} file hashes; {count} completed Validation prediction sets. Test was not read.")
if __name__=="__main__":main()
