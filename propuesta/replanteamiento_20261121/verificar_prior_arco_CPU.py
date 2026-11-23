"""Direct fixed-parameter TT invariance and genuine geometryprior change, CPUonly."""
from base_replanteo import *
import torch,importlib.util
from modelo_prior_arco_subtrayectos import ArcPriorAttention
# Load the original pureNN class in a registered module, withoutcompiledfunctions.
spec=importlib.util.spec_from_file_location('original_time_source_model',OUT/'atencion_subtrayectos_s0/fuentes/modelo_atencion_subtrayectos.py')
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
class Root(torch.nn.Module):
 def __init__(self):
  super().__init__();self.duration=torch.nn.Module();self.duration.register_buffer('log_edges',torch.arange(67.5,1785,15).log())
torch.set_num_threads(2);torch.manual_seed(922)
old=module.SubtrajectoryAttention(Root(),np.zeros(34),np.ones(34),[1.67,.736])
new=ArcPriorAttention(Root(),np.zeros(34),np.ones(34),[1.67,.736])
with torch.no_grad():
 for p in old.branch_parameters():p.add_(.01*torch.randn_like(p))
new.load_state_dict(old.state_dict(),strict=True);old.eval();new.eval()
hidden=torch.randn(5,16,512);raw=torch.randn(5,64,34);raw[:,:,16]=6+.3*torch.randn(5,64)
raw[:,:,27]=(torch.arange(64)[None]<torch.tensor([0,1,5,32,64])[:,None]).float()
geometry=torch.randn(5,64,14);changed=geometry+2
a=old.branch_output(hidden,raw);b=new.branch_output(hidden,raw,geometry);c=new.branch_output(hidden,raw,changed)
for k in [0,2,3,4,5]:assert torch.equal(a[k],b[k]) and torch.equal(b[k],c[k])
assert not torch.equal(a[1],b[1]) and not torch.equal(b[1],c[1])
# Pair canonicalization includes the same sourcepermutation for its geometry.
perm=torch.randperm(64);p=new.branch_output(hidden,raw[:,perm],geometry[:,perm])
for x,y in zip(b,p):assert torch.equal(x,y)
single=new.branch_output(hidden[:1],raw[:1],geometry[:1])
for x,y in zip(b,single):assert torch.allclose(x[:1],y,atol=2e-6,rtol=2e-6) if x.dtype!=torch.bool else torch.equal(x[:1],y)
result=dict(passed=True,scope='Actualattention branch with5queries16members64sources; perturbednonzeroNNparameters, not azeroheadshortcut.',
 same_neural_state_layout=True,all_TTweights_probabilities_gates_sigmas_exact_forfixedparameters=True,
 geometryprior_actually_changes=True,paired_source_permutation_alloutputs_exact=True,
 singleton_tolerance=2e-6,time_source_input_and_normalizer_unchanged=True,
 source_sha256=sha(__file__),model_sha256=sha(OUT/'modelo_prior_arco_subtrayectos.py'),Test_access=False)
write(OUT/'verificacion_prior_arco_CPU.json',result);print(json.dumps(result),flush=True)
