"""Train-only probability, information, fallback and gradient instruments."""
from base_replanteo import *
import torch,copy
from inferencia import feature_input
from modelo_mezcla_rutas import make_duration_model
from modelo_mezcla_proyeccion import ProjectionMixture
from modelo_aux_subviajes import duration_score
access=deny_test();start=time.time();torch.set_num_threads(2);torch.manual_seed(20261122)
data=OUT/'banco_proyeccion_incierta_s0';ds=read(data/'resultado.json')
assert ds['all_primary_rows_retained'] and not ds['Validation_labels_accessed'] and not ds['Test_access']
norm=dict(np.load(data/'normalizacion_Train.npz'));assert sha(data/'normalizacion_Train.npz')==ds['normalization_sha256']
rec=read(BEST/'resultado.json');ck=torch.load(BEST/'modelo.pt',map_location='cpu',weights_only=False)
duration=make_duration_model(ck,rec['config']);duration.load_state_dict(ck['state']);duration.eval()
model=ProjectionMixture(duration,norm['mean'],norm['std']).eval()
tr,X,names=feature_input('train',rec);bank=dict(np.load(data/'train_banco.npz'))
assert np.array_equal(bank['row'],tr['row'])
ii=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index'][:4]
x=torch.from_numpy(X[ii]);C=torch.from_numpy(bank['bank'][ii]);t=torch.from_numpy(np.load(DATA/'train_y.npy')[ii])
with torch.no_grad():
 original=duration(x)[0];base=model.base_only(x)[0]
 out=model(x,C);normerr=float((out[0].exp().sum(-1)-1).abs().max())
 empty=model(x,torch.zeros_like(C))[0];fallback_error=float((empty-original).abs().max())
 perm=[2,3,0,1,6,7,4,5];other=model(x,C[:,perm])
 permutation_error=float((out[0]-other[0]).abs().max())
assert torch.equal(original,base) and normerr<1e-5 and fallback_error<1e-5 and permutation_error<1e-4
assert torch.equal(duration.quantiles(out[0]),duration.quantiles(other[0]))
loss=duration_score(model(x,C)[0],t,duration.duration_grid);assert torch.isfinite(loss);loss.backward()
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
assert model.candidate_head[-1].weight.grad.norm()>0 and model.gate[-1].weight.grad.norm()>0
model.zero_grad(set_to_none=True)
for p in duration.parameters():p.requires_grad_(False)
loss=duration_score(model(x,C)[0],t,duration.duration_grid);loss.backward()
assert all(p.grad is None for p in duration.parameters()) and model.candidate_head[-1].weight.grad.norm()>0
control=ProjectionMixture(copy.deepcopy(duration),norm['mean'],norm['std'],mode='defaults').eval()
control.load_state_dict(model.state_dict())
with torch.no_grad():
 a=control(x,C)[0];perturbed=C.clone();perturbed[:,2:]=C.flip(0)[:,2:];b=control(x,perturbed)[0]
assert torch.equal(a,b)
r=dict(passed=True,scope='Train-only structural instrument, not primary MAE',base_only_exact=True,
 probability_normalization_error=normerr,all_missing_route_fallback_max_logprob_error=fallback_error,
 same_kind_candidate_permutation_max_logprob_error=permutation_error,candidate_permutation_quantiles_exact=True,
 all_gradients_finite=True,candidate_head_gradient_nonzero=True,gate_gradient_nonzero=True,
 frozen_backbone_has_no_gradients=True,default_control_cannot_use_extra_candidates=True,
 information=['o_lon','o_lat','d_lon','d_lat','timestamp'],initial_new_mixture_is_not_original_reference=True,
 model_source_sha256=sha(OUT/'modelo_mezcla_proyeccion.py'),data_manifest_sha256=sha(data/'resultado.json'),
 Validation_data_arrays_accessed=False,Test_access=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(OUT/'verificacion_mezcla_proyeccion.json',r)
print(json.dumps({k:r[k] for k in ['passed','probability_normalization_error','all_missing_route_fallback_max_logprob_error','same_kind_candidate_permutation_max_logprob_error','seconds']}),flush=True)
