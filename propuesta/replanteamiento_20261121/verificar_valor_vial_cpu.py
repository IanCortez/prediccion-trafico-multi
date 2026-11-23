"""Strict CPU publicOD invariants before any primary training."""
from base_replanteo import *
import torch
from modelo_valor_vial import RoadValueNet
from inferencia import feature_input
access=deny_test();start=time.time();torch.set_num_threads(2);torch.manual_seed(20261122)
folder=OUT/'banco_valor_vial_s0';meta=read(folder/'resultado.json');g=dict(np.load(folder/'grafo_estatico.npz'));norm=dict(np.load(folder/'normalizacion_Train.npz'))
assert sha(folder/'grafo_estatico.npz')==meta['static_graph_sha256']
d,X,names=feature_input('train',read(BEST/'resultado.json'));keep=np.asarray([not n.startswith('mm_') for n in names]);x=torch.tensor(X[:17,keep]);del X
z=dict(np.load(folder/'train_proyecciones.npz'));keys=['chain','fraction','error_m','remaining_km','reachable'];bank={k:torch.tensor(z[k][:17].astype(np.int64) if k=='chain' else z[k][:17]) for k in keys};del z
model=RoadValueNet(norm['mean'],norm['std'],g['static'],g['neighbors'],g['neighbor_valid']);model.eval()
with torch.no_grad():
 p=model.public_logprob(x,**bank);assert torch.isfinite(p).all() and torch.allclose(p.exp().sum(-1),torch.ones(17,4),atol=2e-6)
 order=torch.arange(16,-1,-1);r=model.public_logprob(x[order],**{k:v[order] for k,v in bank.items()})[order]
 singleton=torch.cat([model.public_logprob(x[i:i+1],**{k:v[i:i+1] for k,v in bank.items()}) for i in range(17)])
 # Floating arithmetic may vary by matrix kernel. Quantiles must match exactly, probabilities to numeric tolerance.
 assert torch.equal(model.quantiles(p),model.quantiles(r)) and torch.equal(model.quantiles(p),model.quantiles(singleton))
 maxdiff=float((p.exp()-singleton.exp()).abs().max());assert maxdiff<2e-6
 missing={k:v[:1].clone() for k,v in bank.items()};missing['reachable'][:]=False
 fallback=model.public_logprob(x[:1],**missing);assert torch.isfinite(fallback).all() and torch.allclose(fallback.exp().sum(-1),torch.ones(1,4),atol=2e-6)
model.train();p,h=model.public_logprob(x,**bank,return_hidden=True);target=torch.tensor(np.load(DATA/'train_y.npy')[:17]);labels=((target-60)/15).long()
loss=-p.gather(-1,labels[:,None,None].expand(-1,4,1)).mean();loss.backward()
assert all(torch.isfinite(q.grad).all() for q in model.parameters() if q.grad is not None)
result=dict(passed=True,Train_only=True,n=17,probability_normalization=True,batch_reverse_singleton_quantiles_exact=True,
 max_singleton_probability_error=maxdiff,no_reachable_pair_fallback_finite=True,finite_primary_gradients=True,
 public_API='OnlyrawODderivedpublicfeatures/projections accepted, no route/length/durationtargets.',parameters=sum(q.numel() for q in model.parameters()),
 model_sha256=sha(OUT/'modelo_valor_vial.py'),bank_manifest_sha256=sha(folder/'resultado.json'),
 Validation_access=False,Test_access=False,seconds=time.time()-start)
write(OUT/'verificacion_valor_vial_cpu.json',result);print(json.dumps(result),flush=True)
