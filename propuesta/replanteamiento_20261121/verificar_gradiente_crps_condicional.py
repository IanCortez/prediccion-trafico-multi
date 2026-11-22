from base_replanteo import *
from gradiente_crps_condicional import conditional_crps_weights,conditional_crps_surrogate
import torch,itertools
torch.set_num_threads(2)
D=torch.float64
logits=torch.tensor([-.4,.2,.8],dtype=D,requires_grad=True)
unconditional=logits.softmax(-1)
conditional=logits[1:].softmax(-1)
support=torch.tensor([120.,900.],dtype=D)
truth=torch.tensor(450.,dtype=D)
a=(support-truth).abs()
b=(support[:,None]-support[None,:]).abs()
exact=(conditional*a).sum()-.5*(conditional[:,None]*conditional[None,:]*b).sum()
exact_gradient=torch.autograd.grad(exact,logits,retain_graph=True)[0]
assert abs(float(exact_gradient[0]))<1e-14
results=[]
for P in [3,4,5,8]:
    outcome=torch.tensor(list(itertools.product([0,1],repeat=P)),dtype=torch.long)
    times=support[outcome]
    target=truth.expand(len(times))
    weights,scores=conditional_crps_weights(times,target)
    assert torch.allclose(weights.sum(-1),torch.zeros(len(times),dtype=D),atol=1e-12,rtol=0)
    mass=conditional.detach()[outcome].prod(-1)
    assert abs(float(mass.sum())-1)<1e-13
    score_vector=torch.nn.functional.one_hot(outcome+1,3).to(D)-unconditional.detach()[None,None,:]
    expected=((weights[:,:,None]*score_vector).sum(1)*mass[:,None]).sum(0)
    error=float((expected-exact_gradient).abs().max())
    assert error<1e-10
    score_error=abs(float((scores*mass).sum()-exact.detach()))
    assert score_error<1e-10
    logp=logits.log_softmax(-1)[outcome+1]
    surrogate=(weights.detach()*logp).sum(-1)
    g=torch.autograd.grad((surrogate*mass).sum(),logits,retain_graph=True)[0]
    assert torch.allclose(g,exact_gradient,atol=1e-10,rtol=0)
    rev=torch.arange(P-1,-1,-1)
    wr,_=conditional_crps_weights(times[:,rev],target)
    assert torch.allclose(wr,weights[:,rev],atol=1e-12,rtol=0)
    shifted,_=conditional_crps_weights(times+133,target+133)
    doubled,_=conditional_crps_weights(times*2,target*2)
    assert torch.allclose(shifted,weights,atol=1e-12,rtol=0)
    assert torch.allclose(doubled,weights*2,atol=1e-12,rtol=0)
    results.append(dict(particles=P,enumerated_joint_outcomes=len(times),expected_gradient_max_error=error,expected_CRPS_max_error=score_error,gradient_of_inadmissible_category=float(expected[0])))
try:
    conditional_crps_weights(torch.ones(1,2),torch.ones(1))
    raise RuntimeError('P2 must not be accepted')
except AssertionError:pass
same=torch.full((4,8),120.,dtype=D)
w,_=conditional_crps_weights(same,torch.full((4,),450.,dtype=D))
assert torch.count_nonzero(w)==0
assert not torch.cuda.is_initialized()
report=dict(passed=True,synthetic_only=True,exact_enumeration=results,exact_conditional_CRPS=float(exact.detach()),exact_gradient=exact_gradient.detach().numpy(),conditioning='Unconditional category0 is outside the cohort; categories1and2 are eligible. Its gradient cancels exactly although accepted-sample log probabilities remain unconditional.',requirements=['IID complete path/time samples from exact cohort rejection, not normalized finiteCDF particles','At least3accepted samples per query','Full unconditional path log probabilities','No query label in sampling or rejection; fixed60..1785 cohort only','Centering coefficients sum0; unknown cohort-normalizer score cancels'],limitations='Not a trained Porto model, no MAE result, and not yet a verified recurrent trajectory sampler. Exact estimator verification does not prove low training variance or target achievement.',no_GPU=True,Test_access=False,code_sha256=sha(__file__),estimator_sha256=sha(OUT/'gradiente_crps_condicional.py'),references=['https://www.jmlr.org/papers/v21/19-346.html','https://www.jmlr.org/papers/volume25/23-0038/23-0038.pdf','https://www.nature.com/articles/s44387-026-00073-7'])
write(OUT/'verificacion_gradiente_crps_condicional.json',report)
print(json.dumps(report,default=lambda x:x.tolist()))
