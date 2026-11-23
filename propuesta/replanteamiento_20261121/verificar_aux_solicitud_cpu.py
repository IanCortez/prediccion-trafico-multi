"""Exact factorization and real-Train CPU gradient gate, no parameter updates."""
from base_replanteo import *
import torch,inspect
from inferencia import feature_input
from modelo_mezcla_rutas import make_duration_model
from modelo_aux_solicitud import RequestAuxiliary
access=deny_test();torch.set_num_threads(2);torch.manual_seed(17);torch.set_float32_matmul_precision('highest')
start=time.time();rec=read(BEST/'resultado.json');ck=torch.load(BEST/'modelo.pt',map_location='cpu',weights_only=False)
duration=make_duration_model(ck,rec['config']);duration.load_state_dict(ck['state']);duration.eval()
model=RequestAuxiliary(duration);model.eval()
d,X,names=feature_input('train',rec);assert names==rec['features'] and all(not n.startswith('call_') for n in names)
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index']
ii=np.sort(np.random.default_rng(17).choice(eligible,32,replace=False))
x=torch.from_numpy(X[ii].copy());y=torch.from_numpy(np.load(DATA/'train_y.npy')[ii])
c=torch.from_numpy(np.searchsorted(np.array(['A','B','C']),d['call'][ii]).astype(np.int64))
del X,d
out,logq=model(x)
uniform_loss=model.request_score(out,logq,y,c)
assert abs(float(uniform_loss.detach())-float(np.log(3)))<1e-5
with torch.no_grad():
 plain=duration(x)[0];assert torch.equal(out[0],plain)
 # Marginalization over type must exactly recover each duration mixture.
 joint=torch.logsumexp(out[1][...,None,None]+out[4][...,None]+logq[...,None,:],dim=-3)
 marginal_type=torch.logsumexp(joint,-1)
 err=float((marginal_type.exp()-out[0].exp()).abs().max());assert err<1e-6
uniform_loss.backward()
head_grad=float(model.request_head.weight.grad.norm())
assert head_grad>0 and torch.isfinite(model.request_head.weight.grad).all()
# Nonuniform type association must deliver finite nonzero gradients to duration heads.
model.zero_grad(set_to_none=True)
with torch.no_grad():model.request_head.weight.normal_(0,.01)
out,logq=model(x);loss=model.request_score(out,logq,y,c);loss.backward()
base_norm=float(duration.backbone.output.weight.grad.norm())
assert base_norm>0 and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
# Match the conditional probability against direct positive-probability arithmetic.
B,K,J=out[1].shape;label=((y-60)/15).long()
pcomp=out[4].exp().gather(-1,label[:,None,None,None].expand(B,K,J,1))[...,0]
pw=out[1].exp();pc=logq.exp().gather(-1,c[:,None,None,None].expand(B,K,J,1))[...,0]
direct=-((pw*pcomp*pc).sum(-1)/(pw*pcomp).sum(-1)).log().mean()
assert abs(float((direct-loss).detach()))<1e-5
assert list(inspect.signature(RequestAuxiliary.forward).parameters)==['self','x']
r=dict(passed=True,scope='CPU mathematical/information/gradient instrument; no optimizer step and no primary MAE result',
 query_forward='x only, 216 legal OD/map/calendar inputs',
 uniform_type_conditional_NLL=float(uniform_loss.detach()),expected_uniform_NLL=float(np.log(3)),
 type_marginalization_max_probability_error=err,initial_duration_output_bit_identical=True,
 request_head_gradient_norm=head_grad,duration_output_gradient_norm_after_nonuniform_types=base_norm,
 conditional_probability_matches_direct_arithmetic=True,
 actual_request_only_training_target=True,Test_access=False,Validation_access=False,
 actual_files_accessed=sorted(access),parent_sha256=sha(BEST/'modelo.pt'),
 sources={f:sha(OUT/f) for f in ['modelo_aux_solicitud.py','verificar_aux_solicitud_cpu.py']},seconds=time.time()-start)
write(OUT/'verificacion_aux_solicitud_cpu.json',r);print(json.dumps(r),flush=True)
