from base_replanteo import *
from nucleo_entorno_compartido import shared_environment_arrival,conditional_cohort
from nucleo_fases import phase_arrival,geometric_wait_pmf
from nucleo_direccional import directional_arrival
import torch,math
torch.set_num_threads(4)
torch.manual_seed(91)
D=torch.float64
H=24
nodes=torch.tensor([-1.,1.],dtype=D)
prior=torch.tensor([.35,.65],dtype=D)
policy=torch.ones(1,3,1,dtype=D)
weights=torch.ones(1,3,1,dtype=D)
hazards=torch.full_like(weights,.5)
nb=torch.tensor([[1],[2],[2]])
origin=torch.tensor([0])
ends=torch.tensor([[2]])
e=torch.ones(1,1,dtype=D)
ew=torch.zeros(1,1,H,dtype=D)
ew[:,:,0]=1
location=torch.tensor(.1,dtype=D,requires_grad=True)
sigma=torch.tensor(.9,dtype=D,requires_grad=True)
raw=shared_environment_arrival(policy,weights,hazards,[0],ew,e,nb,origin,ends,location,sigma,nodes,prior)
slow=[]
marginal_wait=torch.zeros(1,3,H-1,dtype=D)
for node,pr in zip(nodes,prior):
    hh=(torch.logit(hazards)-location-math.sqrt(2)*sigma*node).sigmoid()
    wait=geometric_wait_pmf(weights,hh,[0],H)
    slow.append(directional_arrival(policy,wait,ew,e,nb,origin,ends))
    marginal_wait=marginal_wait+pr*wait
slow=torch.stack(slow,1)
expected=(slow*prior[None,:,None]).sum(1)
error=float((raw-expected).abs().max().detach())
assert error<1e-12
a=torch.autograd.grad(raw[:,4:].sum(),(location,sigma),retain_graph=True)
b=torch.autograd.grad(expected[:,4:].sum(),(location,sigma))
gerror=max(float((x-y).abs()) for x,y in zip(a,b))
assert gerror<1e-12 and all(torch.isfinite(x) for x in a)
independent=directional_arrival(policy,marginal_wait,ew,e,nb,origin,ends)
difference=float((raw-independent).abs().sum().detach())
assert difference>.01
p,m=conditional_cohort(raw,4,H)
wrong=sum(pr*conditional_cohort(value,4,H)[0] for pr,value in zip(prior,slow.unbind(1)))
conditioning_difference=float((p-wrong).abs().sum().detach())
assert conditioning_difference>.01
with torch.no_grad():
    zero=shared_environment_arrival(policy,weights,hazards,[0],ew,e,nb,origin,ends,0.,0.,nodes,prior)
    original=phase_arrival(policy,weights,hazards,[0],ew,e,nb,origin,ends)
    assert torch.allclose(zero,original,atol=1e-12,rtol=0)
    B=3
    pp=policy.expand(B,-1,-1)
    ww=weights.expand(B,-1,-1)
    hh=torch.tensor([.3,.5,.8],dtype=D)[:,None,None].expand(B,3,1)
    ee=e.expand(B,-1);we=ew.expand(B,-1,-1);oo=origin.expand(B);dd=ends.expand(B,-1)
    all_raw=shared_environment_arrival(pp,ww,hh,[0],we,ee,nb,oo,dd,0.,.8,nodes,prior)
    order=torch.tensor([2,0,1])
    reordered=shared_environment_arrival(pp[order],ww[order],hh[order],[0],we[order],ee[order],nb,oo[order],dd[order],0.,.8,nodes,prior)
    assert torch.allclose(all_raw[order],reordered,atol=1e-12,rtol=0)
    for i in range(B):
        one=shared_environment_arrival(pp[i:i+1],ww[i:i+1],hh[i:i+1],[0],we[i:i+1],ee[i:i+1],nb,oo[i:i+1],dd[i:i+1],0.,.8,nodes,prior)
        assert torch.allclose(one,all_raw[i:i+1],atol=1e-12,rtol=0)
    assert torch.all(all_raw>=0) and torch.all(all_raw.sum(-1)<=1+1e-12)
assert not torch.cuda.is_initialized()
report=dict(passed=True,synthetic_only=True,explicit_convolution_max_error=error,gradient_max_error=gerror,shared_vs_independent_L1_difference=difference,wrong_per_node_cohort_normalization_L1_difference=conditioning_difference,sigma_zero_equals_independent_phase=True,query_permutation_and_singleton=True,mass_conserved=True,no_GPU=True,not_MAE_result=True,code_sha256=sha(__file__),kernel_sha256=sha(OUT/'nucleo_entorno_compartido.py'))
write(OUT/'verificacion_llegada_compartida.json',report)
print(json.dumps(report))
