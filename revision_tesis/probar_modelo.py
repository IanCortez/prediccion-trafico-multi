"""Inspect the retained model on a small prepared sample, CPU by default; no raw-query reconstruction."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BEST="od_mezcla8_geometria_conjunta_s0_fp32_v2"
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--n",type=int,default=8)
    args=parser.parse_args()
    assert 1<=args.n<=65
    import torch
    run=ROOT/"propuesta/experimentos"/BEST
    record=json.loads((run/"resultado.json").read_text(encoding="utf-8-sig"))
    path=run/"modelo.pt"
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record["model_sha256"]
    sys.path.insert(0,str(run/"fuentes"))
    from modelo_mezcla_rutas import make_duration_model
    # Only load the published trusted checkpoint after checking its recorded SHA256.
    ck=torch.load(path,map_location="cpu",weights_only=False)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model=make_duration_model(ck,record["config"]).to(args.device)
    model.load_state_dict(ck["state"])
    model.eval()
    with np.load(ROOT/"revision_tesis/ejemplo_65_validacion.npz",allow_pickle=False) as data:
        assert data["names"].tolist()==record["features"]
        X=torch.as_tensor(data["X"][:args.n],device=args.device)
        expected=data["expected_pred"][:args.n]
    with torch.no_grad():
        output=model(X)
        probabilities=output[0].exp().mean(1)
        prediction=model.quantiles(output[0],(.5,))[:,0].cpu().numpy()
    assert torch.isfinite(probabilities).all()
    assert torch.allclose(probabilities.sum(-1),torch.ones(len(X),device=args.device),atol=1e-5)
    assert np.isfinite(prediction).all()
    print(json.dumps(dict(model_id=BEST,parameters=sum(p.numel() for p in model.parameters()),
                         device=args.device,n=args.n,prediction_s=prediction.tolist(),
                         historical_prediction_s=expected.tolist(),
                         max_difference_s=float(np.max(np.abs(prediction-expected))),
                         full_Validation_not_evaluated=True,Test_not_read=True),indent=2))
if __name__=="__main__":main()
