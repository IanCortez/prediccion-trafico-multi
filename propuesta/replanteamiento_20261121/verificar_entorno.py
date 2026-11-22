from base_replanteo import *
from verosimilitud_entorno import environment_log_likelihood as ll
import torch,math
torch.set_num_threads(2);torch.manual_seed(2);D=torch.float64;nodes,weights=np.polynomial.hermite.hermgauss(4);nodes=torch.tensor(nodes,dtype=D);weights=torch.tensor(weights/math.sqrt(math.pi),dtype=D);lw=torch.randn(3,4,2,dtype=D).log_softmax(-1);haz=torch.randn(3,4,2,dtype=D);duration=torch.tensor([[1,2,3,6],[4,5,1,1],[2,3,4,5]]);valid=torch.tensor([[1,1,1,1],[1,1,0,0],[1,1,1,0]],dtype=torch.bool);lags=torch.tensor([0,1]);loc=torch.tensor(.1,dtype=D,requires_grad=True);zero=torch.tensor(0.,dtype=D)
a=ll(lw,haz,duration,valid,lags,loc,zero,nodes,weights,'shared');b=ll(lw,haz,duration,valid,lags,loc,zero,nodes,weights,'independent');assert (a-b).abs().max()<1e-10
sigma=torch.tensor(.3,dtype=D,requires_grad=True);shared=ll(lw,haz,duration,valid,lags,loc,sigma,nodes,weights,'shared');permutation=torch.tensor([2,0,3,1]);permuted=ll(lw[:,permutation],haz[:,permutation],duration[:,permutation],valid[:,permutation],lags,loc,sigma,nodes,weights,'shared');assert (shared-permuted).abs().max()<1e-10
changed=duration.clone();changed[~valid]=118;assert torch.equal(shared,ll(lw,haz,changed,valid,lags,loc,sigma,nodes,weights,'shared'))
grad=torch.autograd.grad(shared.sum(),[loc,sigma]);assert all(torch.isfinite(g).all() for g in grad)
h=torch.sigmoid(-math.sqrt(2)*.5*nodes);joint=float((weights*h*h).sum());product=float((weights*h).sum()**2);assert joint>product
write(OUT/'verificacion_entorno_cpu.json',dict(passed=True,sigma_zero_models_equal=True,run_order_invariant=True,masked_duration_intervention_invariant=True,finite_location_sigma_gradients=True,shared_geometric_positive_dependence=joint-product,Test_access=False,code_sha256=sha(__file__),likelihood_code_sha256=sha(OUT/'verosimilitud_entorno.py')))
print('Shared/independent likelihood controls and gradients passed.')

