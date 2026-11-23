"""Train-only numerical and information-interface checks, not predictive validation."""
from base_replanteo import *
import torch
from inferencia import feature_input
from inferencia_historica_fresca import load_frozen_history
from modelo_mediana_directa import DirectMedianReadout
access=deny_test();start=time.time();torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.manual_seed(20261122)
reference,ck,r,run,library=load_frozen_history('historica_fresca_s0',device='cpu')
model=DirectMedianReadout(reference).eval()
d,X,names=feature_input('train',read(BEST/'resultado.json'));ii=np.arange(17)
C=np.load(OUT/'banco_proyeccion_incierta_s0/train_banco.npz')['bank'][ii]
H=np.load(OUT/'datos_prototipos_historicos_s0/train_historicos.npz')['history'][ii]
x=torch.tensor(X[ii]);c=torch.tensor(C);h=torch.tensor(H);del X
with torch.no_grad():
 point=model.reference_point(x,c,h);q=model(x,c,h)
 assert torch.equal(q,point[:,None].expand(-1,16))
 assert torch.equal(model.quantize(q.mean(1)),point)
 reverse=model(x.flip(0),c.flip(0),h.flip(0))
 assert torch.equal(q,reverse.flip(0))
 assert all(p.grad is None and not p.requires_grad for p in model.reference.parameters())
model.set_student_training(True);model.train()
# A nonzero deterministic readout fixture exposes the direct gradient path.
with torch.no_grad():model.readout[-1].weight.normal_(0,.001)
members,out=model.training_forward(x,c,h,point.detach())
loss=(members-(point+15)[:,None]).abs().mean()/600;loss.backward()
head_norm=sum(float(p.grad.abs().sum()) for p in model.readout.parameters() if p.grad is not None)
student_norm=sum(float(p.grad.abs().sum()) for p in model.student.parameters() if p.grad is not None)
assert head_norm>0 and student_norm>0
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
assert all(p.grad is None for p in model.reference.parameters())
# Removing the cache must recover exactly the same immutable reference path.
model.eval()
with torch.no_grad():
 cached=model.compute(x,c,h,point.detach())[0];uncached=model(x,c,h)
 assert torch.equal(cached,uncached)
assert not any(Path(q).name.startswith('validation') for q in access)
result=dict(passed=True,scope='17Train inputs only; no predictive improvement claimed.',
 source_sha256=sha(OUT/'modelo_mediana_directa.py'),verification_source_sha256=sha(__file__),
 reference_sha256=r['model_sha256'],zero_initialized_readout_exact_reference=True,
 immutable_reference_no_grad=True,head_gradient_l1=head_norm,student_gradient_l1=student_norm,
 finite_gradients=True,cache_and_uncached_paths_exact=True,zero_readout_reverse_exact=True,
 query=['o_lon','o_lat','d_lon','d_lat','timestamp'],target_as_forward_input=False,
 precision='FP32 highest; direct point residual, no implicit CDF median derivative',
 Validation_access=False,Test_access=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(OUT/'verificacion_mediana_directa_cpu.json',result)
print(json.dumps(dict(passed=True,head_gradient_l1=head_norm,student_gradient_l1=student_norm,seconds=result['seconds'])),flush=True)
