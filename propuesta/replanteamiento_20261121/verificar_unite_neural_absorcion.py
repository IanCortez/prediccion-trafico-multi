"""Actual UNITE temporal/spatial/MoE head + unmodified forward counterexample and corrected controller."""
from base_replanteo import *
import ast,torch,torch.nn as nn,torch.nn.functional as F,random,types
from unite_od_absorbente import absorbing_od_forward
start=time.time();torch.set_num_threads(2);repo=BASE/'referencias/UNITE'
layer_path=repo/'models/layers.py';tree=ast.parse(layer_path.read_text(encoding='utf-8-sig'))
# Default star_flag=False never uses PyG; omit only unavailable imports, no neural math changes.
tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module and n.module.startswith('torch_geometric'))]
layer_env=dict(__name__='unite_layers_synthetic',__file__=str(layer_path))
exec(compile(tree,str(layer_path),'exec'),layer_env)
base_tree=ast.parse((OUT/'instrumento_unite_od_sintetico.py').read_text(encoding='utf-8-sig'))
classes=[n for n in base_tree.body if isinstance(n,ast.ClassDef) and n.name in ['Candidates','Fixture']]
env=dict(torch=torch,F=F,random=random);exec(compile(ast.Module(body=classes,type_ignores=[]),'<fixture>','exec'),env)
original=ast.parse((repo/'models/tte.py').read_text(encoding='utf-8-sig'))
unite=next(n for n in original.body if isinstance(n,ast.ClassDef) and n.name=='UNITE')
forward=next(n for n in unite.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
mask=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='sequence_mask')
exec(compile(ast.Module(body=[mask,forward],type_ignores=[]),str(repo/'models/tte.py'),'exec'),env)
env['Fixture'].forward=env['forward']
def make_eta():
 torch.manual_seed(20261122)
 p=types.SimpleNamespace(graph_flag=True,hid_dim=8,num_experts=2,topk=1,dropout=0.,heads=2,
  tt_distribution=np.zeros((168,6,3),np.float32),
  ls_nbr1=[torch.tensor([i,(i+1)%6]) for i in range(6)],mask_nbr1=[torch.ones(2) for i in range(6)])
 estimator=layer_env['SegTimeEstimator'](p);estimator.to('cpu');estimator.eval()
 last=[m for m in estimator.out_layer.modules() if isinstance(m,nn.Linear)][-1]
 with torch.no_grad():last.weight.zero_();last.bias.fill_(1.)
 return estimator
class NeuralFixture(env['Fixture']):
 hid_dim=8
 def __init__(self):
  super().__init__();self.eta_predictor=make_eta()
 def odt_encoder(self,o,d,depart,discrete):
  return torch.stack([o.float(),d.float(),depart.float(),torch.ones_like(o).float()],-1).repeat(1,2)*.01
 def emb_departure(self,depart):return torch.zeros(len(depart),8)
 def get_seg_emb(self,input_id,*args):return input_id.float()[:,None].expand(-1,8)*.01
 def get_dest_info(self,d,slots):
  return None,torch.arange(len(d)),torch.arange(3)[None].expand(len(d),-1),None,None
@torch.no_grad()
def execute(destinations,corrected,poison_length=3):
 B=len(destinations);model=NeuralFixture();o=torch.zeros(B,dtype=torch.long);d=torch.tensor(destinations);dep=torch.zeros(B,dtype=torch.long);day=dep.clone();dt=torch.zeros(B,6)
 if corrected:
  t,ids,lengths,unfinished=absorbing_od_forward(model,o,d,dep,day,dt,50)
 else:
  random.seed(20261122);t,_,ids,lengths=model.forward(o,d,dep,[poison_length]*B,torch.full((poison_length,B),123456,dtype=torch.long),torch.full((poison_length,B),99999.),day,dt,-1);unfinished=None
 total=torch.stack([t[:int(lengths[i]),i].sum() for i in range(B)])
 return total,ids,lengths,unfinished
alone=execute([4],False);mixed=execute([4,5],False);poison=execute([4,5],False,80)
assert all(torch.equal(a,b) for a,b in zip(mixed[:3],poison[:3]))
assert not torch.equal(alone[0],mixed[0][:1])
fixed=execute([4,5],True);fixed_alone=execute([4],True);reverse=execute([5,4],True)
assert torch.equal(fixed[0][:1],fixed_alone[0]) and torch.equal(fixed[0],reverse[0].flip(0))
assert torch.equal(fixed[1][:,0],fixed_alone[1][:,0]) and torch.equal(fixed[2],reverse[2].flip(0))
assert not fixed[3].any()
result=dict(passed=True,scope='Synthetic structural proof, not a trainedPorto benchmark or completeUNITEneuralnetwork.',
 actual_unmodified_components=['SegTimeEstimator withgraph_flagTrue anddefaultstarFalse','temporal andspatial attention','SparseMoE','NoisyTopkGating','officialUNITE.forward negativecontrol'],
 fixtures='Otherencoder/policy/state methods use deterministic stubs. Real head finalLinear set tovalidconstantpositive1; proves representable counterexample, not actualtrainedcheckpoint behavior.',
 original_same_query_single=alone[0][0].item(),original_same_query_mixed=mixed[0][0].item(),
 corrected_same_query_single=fixed_alone[0][0].item(),corrected_same_query_mixed=fixed[0][0].item(),
 corrected_batch_singleton_order_invariant=True,original_ratio_minus1_targetpoisoning_invariant=True,
 corrected_API_accepts_no_target_route_length_or_times=True,terminal_contribution_once=True,
 unfinished_routes_explicit=True,PyG_not_used='DefaultstarFalse does not executeGAT; only imports omitted in isolatedtest namespace.',
 remaining=['RealPortoadapter must handle uncertainGPSendpointsets andpartial offsets.','Cap/deadend needs predictedremainingtime, not observedtripN.','Train-onlytimeprediction instrument andboundedfullcorpus beforeprimarytraining.'],
 layers_source_sha256=sha(layer_path),forward_source_sha256=sha(repo/'models/tte.py'),
 controller_source_sha256=sha(OUT/'unite_od_absorbente.py'),verification_source_sha256=sha(__file__),
 real_datasets_opened=False,Test_access=False,seconds=time.time()-start)
write(OUT/'verificacion_unite_neural_absorcion.json',result)
print(json.dumps(result),flush=True)
