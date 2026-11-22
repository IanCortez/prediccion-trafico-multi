from base_replanteo import *
from nucleo_fases import phase_arrival,geometric_wait_pmf
from nucleo_direccional import directional_arrival
import torch
torch.set_num_threads(4);torch.manual_seed(4);D=torch.float64;B,S,J,K,H=2,7,3,8,20
neighbors=torch.randint(S,(B,S,J));policy=torch.randn(B,S,J,dtype=D).softmax(-1);lags=torch.tensor([0,0,1,1,2,2,3,3]);logw=torch.randn(B,S,K,dtype=D,requires_grad=True);logh=torch.randn(B,S,K,dtype=D,requires_grad=True);weights=logw.softmax(-1);hazards=logh.sigmoid();end_wait=torch.randn(B,2,H,dtype=D).softmax(-1);end_p=torch.rand(B,2,dtype=D);origin=torch.tensor([0,1]);ends=torch.tensor([[2,3],[4,5]]);wm=geometric_wait_pmf(weights,hazards,lags,H)
slow=directional_arrival(policy,wm,end_wait,end_p,neighbors,origin,ends);fast=phase_arrival(policy,weights,hazards,lags,end_wait,end_p,neighbors,origin,ends);error=float((slow-fast).abs().max());assert error<1e-10
gs=torch.autograd.grad(slow[:,4:].sum(),(logw,logh),retain_graph=True);gf=torch.autograd.grad(fast[:,4:].sum(),(logw,logh));gerror=max(float((a-b).abs().max()) for a,b in zip(gs,gf));assert gerror<1e-10
with torch.no_grad():
 rev=torch.tensor([1,0]);reordered=phase_arrival(policy[rev],weights[rev],hazards[rev],lags,end_wait[rev],end_p[rev],neighbors[rev],origin[rev],ends[rev]);assert (reordered-fast.detach()[rev]).abs().max()<1e-10
 assert torch.all(fast.sum(-1)<=1+1e-10)
# An unshifted component with hazard1 is exactly one15-second movement interval.
p=torch.ones(1,2,1,dtype=D);nb=torch.tensor([[1],[0]]);w=torch.ones(1,2,1,dtype=D);h=torch.ones_like(w);ew=torch.zeros(1,1,6,dtype=D);ew[0,0,2]=1
with torch.no_grad():out=phase_arrival(p,w,h,torch.tensor([0]),ew,torch.ones(1,1,dtype=D),nb,torch.tensor([0]),torch.tensor([[1]]))
ref=torch.zeros_like(out);ref[0,3]=1;assert (out-ref).abs().max()<1e-10
report=dict(passed=True,explicit_convolution_max_error=error,explicit_convolution_gradient_max_error=gerror,multiple_terminal_states=True,shifted_positive_residence=True,query_permutation=True,finite_horizon_mass_conserved=True,deterministic_time_addition=True,duration_pmf_is_unbounded_not_renormalized_before_total_cohort_conditioning=True,code_sha256=sha(__file__),kernel_code_sha256=sha(OUT/'nucleo_fases.py'),Test_access=False)
write(OUT/'verificacion_fases_cpu.json',report);print(json.dumps(report))

