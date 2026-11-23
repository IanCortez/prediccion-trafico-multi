"""Stable-gradient CUDA check on earlier fitlabels; frozen modelparameters stay unchanged."""
from base_replanteo import *
import torch,gc
from inferencia import feature_input
from modelo_mezcla_individual_mediana import IndividualSourceMixture
from cuantiles_mezcla_balanceada import stable_mixture_loss
from tokens_transferencia_arco import ArcTransferBank
from datos_mezcla_individual_Train import tensor_tokens
start=time.time();access=deny_test()
torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(.5)
run=OUT/'mezcla_individual_mediana_Train_s0';result=read(run/'resultado.json');assert read(run/'reproduccion_independiente.json')['passed']
with np.load(OUT/'oraculo_tiempo_calle_train_s0/auxiliares_Train.npz') as z:q={k:z[k][:256] for k in ['train_index','true','timestamp']}
d,X,names=feature_input('train',read(BEST/'resultado.json'));keep=np.array([not n.startswith('mm_') for n in names]);X=X[q['train_index']][:,keep]
queries={k:d[k][q['train_index']] for k in ['o_lon','o_lat','d_lon','d_lat','timestamp']}
with np.load(OUT/'piloto_cdf_ordenada_Train_s0/indices_fuentes.npz') as z:refs={k:z[k][:256] for k in ['parent','first','last']}
meta=torch.load(run/'constructor.pt',map_location='cpu',weights_only=False);bank=ArcTransferBank('cuda')
raw=tensor_tokens(bank,queries,X,**refs,shape_indices=meta['shape_indices'],available_indices=meta['available_indices'])
checkpoint=[v for v in result['checkpoints'] if v['arm']=='observed'][0];assert sha(checkpoint['file'])==checkpoint['sha256']
ck=torch.load(checkpoint['file'],map_location='cpu',weights_only=False);model=IndividualSourceMixture(**ck['constructor']).cuda();model.load_state_dict(ck['state']);model.eval()
before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
x=torch.tensor(X,device='cuda');y=torch.tensor(q['true'],device='cuda',dtype=torch.float32)
torch.cuda.reset_peak_memory_stats();loss,mae,nll=stable_mixture_loss(model(x,raw),y)
assert torch.isfinite(loss);loss.backward()
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
assert any(p.grad is not None and float(p.grad.abs().sum())>0 for p in model.members[0].query.parameters())
for k,v in model.state_dict().items():assert torch.equal(v.detach().cpu(),before[k])
proof=dict(passed=True,fit_only_n=256,no_heldlabel_gradient=True,stable_loss=float(loss.detach()),fit_median_MAE_s=float(mae.detach())*600,
 bin_NLL=float(nll.detach()),all_parameter_gradients_finite=True,query_gradient_nonzero=True,
 optimizer_step_executed=False,checkpoint_parameters_unchanged=True,peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,
 stable_solver_sha256=sha(OUT/'cuantiles_mezcla_balanceada.py'),model_sha256=checkpoint['sha256'],
 code_sha256=sha(__file__),Validation=False,Test=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(OUT/'preflight_balanceado_GPU.json',proof);print(json.dumps(proof),flush=True)
