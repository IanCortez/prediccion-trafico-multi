"""Train-only instruments for a fresh joint22component historical route model."""
from base_replanteo import *
import torch,copy
from inferencia import feature_input
from modelo_mezcla_rutas import make_duration_model
from modelo_mezcla_proyeccion import ProjectionMixture
from modelo_mezcla_historica import HistoricalRouteMixture
from modelo_aux_subviajes import duration_score
access=deny_test();start=time.time();torch.set_num_threads(2);torch.manual_seed(0)
fresh=torch.load(OUT/'campo_multiescala_s0/constructor.pt',map_location='cpu',weights_only=False)
assert 'state' not in fresh and fresh['Train_n']==1023485
rec=read(BEST/'resultado.json');assert fresh['features']==rec['features']
mapnorm=dict(np.load(OUT/'banco_proyeccion_incierta_s0/normalizacion_Train.npz'))
data=OUT/'datos_prototipos_historicos_s0';prep=read(data/'resultado.json');fix=read(data/'calibracion_fp64.json')
assert read(data/'verificacion_entrada.json')['passed'] and prep['own_fold_intervention_passed']
assert fix['rank']==2 and fix['normalization_mean_std_unchanged'] and not fix['Test_access']
norm=dict(np.load(data/fix['used_constructor_file']));assert sha(data/fix['used_constructor_file'])==fix['used_constructor_sha256']
duration=make_duration_model(fresh,rec['config']);parent=ProjectionMixture(duration,mapnorm['mean'],mapnorm['std'],'expanded')
model=HistoricalRouteMixture(parent,norm['mean'],norm['std'],norm['time_anchor']).eval()
with torch.no_grad():model.gate[-1].bias.zero_()
tr,X,names=feature_input('train',rec);ii=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index'][:4]
C=torch.from_numpy(np.load(OUT/'banco_proyeccion_incierta_s0/train_banco.npz')['bank'][ii])
H=torch.from_numpy(np.load(data/'train_historicos.npz')['history'][ii]);x=torch.from_numpy(X[ii]);t=torch.from_numpy(np.load(DATA/'train_y.npy')[ii])
with torch.no_grad():
 base=parent(x,C)[0];fallback=model(x,C,torch.zeros_like(H))[0]
 out=model(x,C,H);permuted=model(x,C,H[:,[2,0,5,1,4,3]])
 normerr=float((out[0].exp().sum(-1)-1).abs().max());perr=float((out[0]-permuted[0]).abs().max())
assert out[1].shape[-1]==22 and normerr<1e-5 and perr<1e-4
assert torch.equal(fallback,base) and torch.equal(duration.quantiles(out[0]),duration.quantiles(permuted[0]))
loss=duration_score(model(x,C,H)[0],t,duration.duration_grid);loss.backward()
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
assert model.history_head[-1].weight.grad.norm()>0 and model.gate[-1].weight.grad.norm()>0
model.zero_grad(set_to_none=True)
control=HistoricalRouteMixture(copy.deepcopy(parent),norm['mean'],norm['std'],norm['time_anchor'],mode='parent_control').eval();control.load_state_dict(model.state_dict())
with torch.no_grad():
 a=control(x,C,H)[0];b=control(x,C,H.flip(0))[0]
assert torch.equal(a,b) and torch.equal(a,base)
r=dict(passed=True,scope='Train-only fresh-model probability/information instruments, not primaryMAE',
 fresh_global_weights_no_retained_checkpoint_loaded=True,preprocessing_only_ctor_Train_n=fresh['Train_n'],
 components=22,initial_historical_gate=.5,probability_normalization_error=normerr,historical_set_permutation_logprob_error=perr,
 all_missing_history_parent_fallback_exact=True,permuted_history_quantiles_exact=True,all_gradients_finite=True,
 new_head_and_gate_gradients_nonzero=True,parent_control_history_perturbation_invariant=True,parent_control_exact_parent_prediction=True,
 own_fold_source_intervention_from_preparation_passed=True,physical_calibration_fp64_rank=2,
 model_source_sha256=sha(OUT/'modelo_mezcla_historica.py'),parent_model_source_sha256=sha(OUT/'modelo_mezcla_proyeccion.py'),
 base_constructor_sha256=sha(OUT/'campo_multiescala_s0/constructor.pt'),
 history_constructor_sha256=fix['used_constructor_sha256'],Validation_data_arrays_accessed=False,Test_access=False,
 actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(OUT/'verificacion_mezcla_historica_fresca.json',r)
print(json.dumps({k:r[k] for k in ['passed','components','probability_normalization_error','historical_set_permutation_logprob_error','seconds']}),flush=True)
