"""CPU real-neural identity, gradient and fulltrip-sum objective instrument."""
from base_replanteo import *
import torch
from modelo_progreso_recurrente import ProgressNet
from modelo_consistencia_temporal_viaje import ConsistentProgressNet
deny_test();torch.set_num_threads(2);torch.manual_seed(7951)
g=dict(static=np.zeros((3,12),np.float32),chain_length=np.array([10,20,30],np.float32),
 chain_src=np.arange(3),chain_dst=np.arange(1,4),
 neighbors=np.array([[1,3,3,3,3,3],[2,3,3,3,3,3],[3,3,3,3,3,3]]),
 neighbor_valid=np.array([[1,0,0,0,0,0],[1,0,0,0,0,0],[0,0,0,0,0,0]],bool),
 D=np.array([[0,10,30,60],[np.inf,0,20,50],[np.inf,np.inf,0,30],[np.inf,np.inf,np.inf,0]],np.float32),
 summer_start=np.array([1396141200]),summer_end=np.array([1414285200]))
base=ProgressNet(np.zeros(216),np.ones(216),g).eval()
model=ConsistentProgressNet(np.zeros(216),np.ones(216),g).eval()
missing,unexpected=model.load_state_dict(base.state_dict(),strict=False)
assert set(missing)=={'time_correction.0.weight','time_correction.0.bias','time_correction.2.weight','time_correction.2.bias'} and not unexpected
chain=torch.zeros(2,2,4,dtype=torch.long);chain[1,1]=2
frac=torch.zeros(2,2,4);frac[:,1]=.5
query=dict(x=torch.zeros(2,216),timestamp=torch.tensor([1400000000,1400000001]),chain=chain,fraction=frac,
 error_m=torch.zeros(2,2,4),remaining_km=torch.ones(2,4,4),reachable=torch.ones(2,4,4,dtype=torch.bool))
before=base.public_rollout(**query,max_steps=8)
identity=model.public_rollout(**query,max_steps=8);assert torch.equal(before,identity)
parent_state={k:v.clone() for k,v in model.state_dict().items() if not k.startswith('time_correction.')}
b=dict(chain=torch.tensor([[0,1,2],[0,1,2]]),destination=torch.tensor([2,2]),
 destination_m=torch.tensor([15.,15.]),entry_m=torch.zeros(2,3),distance_m=torch.tensor([[10.,20.,15.],[10.,20.,15.]]),
 timestamp=torch.tensor([1400000000,1400000001]),elapsed_s=torch.tensor([[0.,8.,19.],[0.,6.,18.]]),
 valid=torch.ones(2,3,dtype=torch.bool),local_s=torch.ones(2,3)*100,
 remaining_s=torch.ones(2,3)*999,recognition=torch.ones(2,18)*777)
x=torch.zeros(2,216);table=model.roads(torch.arange(4)).detach()
initial,_=model.private_local_prediction(x,b,table);target=initial.detach().sum(1)*1.35
poison={**b,'local_s':b['local_s']*1000,'remaining_s':b['remaining_s']*-99,'recognition':b['recognition']*0}
other,_=model.private_local_prediction(x,poison,table);assert torch.equal(initial,other)
params=[p for p in model.parameters() if p.requires_grad];assert sum(p.numel() for p in params)==8386
opt=torch.optim.Adam(params,lr=.01);losses=[]
for step in range(35):
 opt.zero_grad(set_to_none=True);local,correction=model.private_local_prediction(x,b,table)
 total=(local*b['valid']).sum(1);loss=torch.abs(total-target).mean()/600+.001*correction.square().mean()
 assert torch.isfinite(loss);loss.backward()
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
 assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('time_correction.'))
 opt.step();losses.append(float(loss.detach()))
 assert torch.all(local>0)
assert min(losses[-5:])<losses[0]*.25,(losses[0],losses[-5:])
for k,v in parent_state.items():assert torch.equal(model.state_dict()[k],v),k
assert 'D' not in model.state_dict() and not torch.cuda.is_initialized()
result=dict(passed=True,CPU_only=True,actual_neural_public_rollout_initially_exact_parent=True,
 wholetrip_aggregate_loss_decreased=True,initial_loss=losses[0],last5_losses=losses[-5:],
 only_correction_gradients=True,frozen_root_policy_GRU_embeddings_unchanged=True,
 private_T_localtargets_recognition_poisoning_cannot_affect_prediction=True,
 public_query_interface_unchanged=True,positive_pace_and_delay=True,new_parameters=8386,
 D_nonpersistent=True,parent_source_sha256=sha(OUT/'modelo_progreso_recurrente.py'),
 model_source_sha256=sha(OUT/'modelo_consistencia_temporal_viaje.py'),source_sha256=sha(__file__),
 synthetic_proof_not_a_performance_score=True,Test=False)
write(OUT/'verificacion_consistencia_temporal_viaje.json',result);print(json.dumps(result),flush=True)
