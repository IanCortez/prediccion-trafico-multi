"""Actual Train gradient smoke, not a fit or a predictive result; CPU only."""
from base_replanteo import *
from modelo_prefijo_nativo import NativePrefixArrival
from muestreador_rechazo_nativo import sample_cohort_trajectories
from gradiente_crps_condicional import conditional_crps_surrogate
from inferencia import feature_input
import torch
torch.set_num_threads(2);torch.manual_seed(2026)
access=deny_test()
assert read(OUT/'verificacion_muestreador_rechazo.json')['passed']
start=time.time()
parent=OUT/'prefijo_nativo_s0'
expected=read(parent/'congelacion_para_inferencia.json')['model_sha256']
assert sha(parent/'modelo.pt')==expected
ck=torch.load(parent/'modelo.pt',map_location='cpu',weights_only=False)
grid=ck['grid']
graph=dict(np.load(OUT/'datos_semi_markov/grafo_grid.npz'))
distance=np.load(OUT/'datos_semi_markov/distancias_grid.npy')
counts=np.load(OUT/'prior_transiciones/conteos_fold.npz')['counts']
model=NativePrefixArrival(ck['stats'],grid,graph,distance,ck['power'],counts,ck['phase_initialization'])
model.load_state_dict(ck['state'])
del ck,counts
# Deterministic query representation; GRU training mode supports eventual CUDA
# backward, with dropout0. Encoder Dropout remains disabled.
model.eval();model.prefix_rnn.train()
tr,X,names=feature_input('train',read(BEST/'resultado.json'))
y=np.load(DATA/'train_y.npy')
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index']
ii=np.random.default_rng(2026).choice(eligible,4,replace=False)
def ids(tag):
    xx=(tr[tag+'_lon'][ii]+8.62)*111195*np.cos(np.deg2rad(41.16))
    yy=(tr[tag+'_lat'][ii]-41.16)*111195
    return torch.tensor(np.floor((xx-grid['x0'])/200).astype(np.int64)*grid['ny']+np.floor((yy-grid['y0'])/200).astype(np.int64))
x=torch.tensor(X[ii],dtype=torch.float32);del X
o=ids('o');d=ids('d')
timestamp=torch.tensor(tr['timestamp'][ii],dtype=torch.int64)
fold=(timestamp//(7*86400))%5
sample=sample_cohort_trajectories(model,x,o,d,particles=4,candidates=8,max_rounds=4,fold=fold,timestamp=timestamp)
complete=sample['complete']
assert complete.any()
target=torch.tensor(y[ii],dtype=torch.float32)
loss,score=conditional_crps_surrogate(sample['times'][complete],target[complete],sample['log_probability'][complete])
assert torch.isfinite(loss) and torch.isfinite(score)
loss.backward()
norms={}
for name,p in model.named_parameters():
    if p.grad is not None:
        assert torch.isfinite(p.grad).all(),name
        if name in ['prefix_rnn.weight_ih_l0','route_head.weight','stay_head.weight','stop_head.weight']:norms[name]=float(p.grad.norm())
assert sum(norms.values())>0
assert not torch.cuda.is_initialized()
assert sha(parent/'modelo.pt')==expected
report=dict(passed=True,scope='Four real Train OD queries, no optimizer step, no predictive MAE; successful gradient propagation only',Train_query_count=4,fully_sampled_queries=int(complete.sum()),samples_per_query=4,sampled_T_seconds=sample['times'].detach().numpy(),rounds=sample['rounds'],candidate_attempts_per_query=sample['candidate_attempts_per_query'],encoder_dropout_disabled=True,GRU_train_mode_with_dropout0=True,route_counts_exclude_own_Train_weekfold=True,query_target_duration_supplied_only_after_sampling=True,gradient_norms=norms,sample_CRPS_scaled_by600=float(score),no_optimizer_or_model_update=True,model_sha256=expected,no_GPU=True,Test_access=False,actual_files_accessed=sorted(access),code_sha256=sha(__file__),sampler_sha256=sha(OUT/'muestreador_rechazo_nativo.py'),seconds=time.time()-start)
write(OUT/'sondeo_riesgo_total_cpu.json',report)
print(json.dumps(report,default=lambda x:x.tolist()))
