"""Exact scoring identity and real-Train gradient gate, no optimization or GPU."""
from base_replanteo import *
import torch,shutil
from scipy.special import ndtr
from inferencia import feature_input
from modelo_mezcla_rutas import make_duration_model
from perdida_maestro_probabilistico import teacher_probability,expected_teacher_score
access=deny_test();torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.manual_seed(17)
start=time.time()
mu=torch.tensor([np.log(70),np.log(600),np.log(1750)],dtype=torch.float64,requires_grad=True)
ls=torch.tensor([np.log(.1),np.log(.3),np.log(.08)],dtype=torch.float64,requires_grad=True)
q=teacher_probability(mu,ls);assert q.shape==(3,116) and torch.max(torch.abs(q.sum(-1)-1))<1e-14
edges=np.log(np.arange(67.5,1785,15));cdf=ndtr((edges[None,:]-mu.detach().numpy()[:,None])/ls.detach().exp().numpy()[:,None])
npq=np.diff(np.column_stack([np.zeros(3),cdf,np.ones(3)]),axis=1).clip(1e-9);npq/=npq.sum(1,keepdims=True)
err=float(np.abs(q.numpy()-npq).max());assert err<1e-13
raw=torch.randn(3,2,116,dtype=torch.float64,requires_grad=True);lp=raw.log_softmax(-1);Fp=lp.exp().cumsum(-1);Fq=q.cumsum(-1)
soft=expected_teacher_score(lp,mu,ls)
grid=torch.arange(116);indicator=(grid[None,:]>=grid[:,None]).double()
hard_rps=(Fp[:,:,None,:]-indicator[None,None,:,:]).square().mean(-1)
expected_hard=(-(q[:,None,:]*lp).sum(-1)+(5*hard_rps*q[:,None,:]).sum(-1)).mean()
constant=5*(Fq*(1-Fq)).mean();assert abs(float((expected_hard-soft-constant).detach()))<1e-12
g1=torch.autograd.grad(expected_hard,raw,retain_graph=True)[0];g2=torch.autograd.grad(soft,raw)[0]
graderr=float((g1-g2).abs().max());assert graderr<1e-12
assert mu.grad is None and ls.grad is None and not q.requires_grad
params=dict(np.load(OUT/'piloto_maestro_probabilistico_s0/held_Train_parametros.npz'))
rec=read(BEST/'resultado.json');tr,X,names=feature_input('train',rec);assert names==rec['features'] and all(not n.startswith('call_') for n in names)
lookup=np.full(int(tr['row'].max())+1,-1,np.int32);lookup[tr['row']]=np.arange(len(tr['row']),dtype=np.int32);ii=lookup[params['row'][:32]];assert (ii>=0).all()
x=torch.from_numpy(X[ii].copy());del X
ck=torch.load(BEST/'modelo.pt',map_location='cpu',weights_only=False);model=make_duration_model(ck,rec['config']);model.load_state_dict(ck['state']);model.eval()
pr=params['privileged_geometry'][:32].astype(np.float32)
out=model(x);loss=expected_teacher_score(out[0],torch.from_numpy(pr[:,0]),torch.from_numpy(pr[:,1]));loss.backward()
assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
norm=float(model.backbone.output.weight.grad.norm());assert norm>0
result=dict(passed=True,scope='CPU proof only; no optimizer step and no primary MAE result',
 scipy_Torch_probability_max_error=err,expected_hard_score_identity_error=float((expected_hard-soft-constant).detach()),expected_hard_score_gradient_max_error=graderr,
 teacher_parameters_detached=True,real_Train_finite_nonzero_output_gradient_norm=norm,
 scalar_loss_for_32_Train_rows=float(loss.detach()),query_model_unchanged=True,
 teacher_geometry_not_a_model_forward_input=True,Validation_access=False,Test_access=False,
 actual_files_accessed=sorted(access),reference_sha256=sha(BEST/'modelo.pt'),
 sources={f:sha(OUT/f) for f in ['perdida_maestro_probabilistico.py','verificar_perdida_maestro.py']},seconds=time.time()-start)
write(OUT/'verificacion_perdida_maestro.json',result);print(json.dumps(result),flush=True)
