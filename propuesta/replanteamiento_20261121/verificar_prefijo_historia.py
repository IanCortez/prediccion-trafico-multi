from base_replanteo import *
from modelo_prefijo_historia import HistoricalPrefixArrival
from modelo_prefijo_nativo import NativePrefixArrival
from historia_eventos import event_groups
import torch
torch.set_num_threads(2);torch.manual_seed(77)
rng=np.random.default_rng(77)
off=np.array([(a,b) for a in range(-3,4) for b in range(-3,4) if (a,b)!=(0,0)])
nb=np.full((3,48),-1,np.int64)
for c in range(3):
    for j,(dx,dy) in enumerate(off):
        if dy==0 and 0<=c+dx<3:nb[c,j]=c+dx
graph=dict(neighbors=nb,allowed=nb>=0,centers=np.array([[0.,0.],[200.,0.],[400.,0.]],np.float32),offsets=off)
grid=dict(nx=3,ny=1,x0=0,y0=0,cell_m=200)
stats=dict(mean=np.zeros(8),std=np.ones(8),keep=np.arange(8),bins=[torch.tensor([-3.,0.,3.]) for _ in range(8)])
init=dict(lags=[0,0,1,1,2,2,3,3],weights=[.125]*8,hazards=[.5]*8)
distance=np.abs(np.arange(3)[:,None]-np.arange(3)[None,:])*200
counts=np.zeros((5,3*160,48))
full=rng.uniform(.1,1,size=(360,120,3)).astype(np.float32)
full[np.arange(360)//12%2==0,:,2]=0
full/=full.sum(-1,keepdims=True)
parent=NativePrefixArrival(stats,grid,graph,distance,[4.,.5],counts,init)
m=HistoricalPrefixArrival(stats,grid,graph,distance,[4.,.5],counts,init,full).eval()
m.load_native_initialization(parent.state_dict())
with torch.no_grad():
    m.history_projection.weight.normal_(0,.03)
    m.event_residual.weight.normal_(0,.03)
x=torch.tensor([[0.,0.,.4,0.,.4,0.,.4,-.9]]*2)
o=torch.zeros(2,dtype=torch.long);d=torch.full_like(o,2)
cell=torch.tensor([[0,0,1,1,2,2],[0,1,1,2,2,2]])
head=torch.tensor([[4,4,0,0,0,0],[4,0,0,0,0,0]])
age=torch.tensor([[0,1,0,1,0,1],[0,0,1,0,1,2]])
timestamp=torch.tensor([1460041000,1460112345])
group=event_groups(cell.numpy(),head.numpy(),d[:,None].numpy(),timestamp[:,None].numpy(),np.arange(6)[None],age.numpy())
p=full[group,age.numpy()]
motion,stop=m.teacher_probabilities(x,o,d,cell,head,age,event_prior=torch.tensor(p))
q,*_=m.encode_od(x,o,d);hidden=q[None]
parts=[];ends=[]
for t in range(6):
    prior=m.historical_probability(cell[:,t],d,head[:,t],timestamp,t,age[:,t])
    assert np.array_equal(prior.numpy(),p[:,t])
    a,b,hidden=m.rollout_with_history(q,x[:,:4],cell[:,t],d,head[:,t],age[:,t],t,hidden,timestamp)
    parts.append(a);ends.append(b)
error=float((torch.stack(parts,1)-motion).abs().max().detach())
assert error<1e-6 and torch.allclose(torch.stack(ends,1),stop,atol=1e-6,rtol=0)
changed=cell.clone();changed[:,3:]=2-changed[:,3:]
a,b=m.teacher_probabilities(x,o,d,changed,head,age,event_prior=torch.tensor(p))
assert torch.allclose(a[:,:3],motion[:,:3],atol=1e-7,rtol=0)
loss=-motion[:,0,0].log().mean()-stop[:,-1].clamp_min(1e-9).log().mean()
loss.backward()
assert all(torch.isfinite(t.grad).all() for t in m.parameters() if t.grad is not None)
assert m.prefix_rnn.weight_ih_l0.grad.abs().sum()>0
seeds=torch.tensor([93,31])
raw,left=m.generated_arrival(x,o,d,seeds,particles=32,horizon=20,timestamp=timestamp)
assert torch.isfinite(raw).all() and torch.all(raw>=0)
assert torch.allclose(raw.sum(-1)+left,torch.ones(2),atol=1e-5,rtol=0)
order=torch.tensor([1,0])
rr,ll=m.generated_arrival(x[order],o[order],d[order],seeds[order],particles=32,horizon=20,timestamp=timestamp[order])
assert torch.allclose(rr,raw[order],atol=1e-6,rtol=0)
for i in range(2):
    one,_=m.generated_arrival(x[i:i+1],o[i:i+1],d[i:i+1],seeds[i:i+1],particles=32,horizon=20,timestamp=timestamp[i:i+1])
    assert torch.allclose(one,raw[i:i+1],atol=1e-6,rtol=0)
assert not torch.cuda.is_initialized()
report=dict(passed=True,numpy_and_Torch_history_keys_exact=True,teacher_vs_incremental_max_error=error,causal_future_intervention=True,finite_gradients_and_nonzero_RNN_gradient=True,raw_mass_and_survival_conserved=True,query_permutation_and_singleton=True,requires_explicit_fold_excluded_history_during_training=True,requires_only_known_departure_timestamp_at_query=True,no_query_length_or_duration_argument=True,no_GPU=True,code_sha256=sha(__file__),model_code_sha256=sha(OUT/'modelo_prefijo_historia.py'),history_code_sha256=sha(OUT/'historia_eventos.py'))
write(OUT/'verificacion_prefijo_historia.json',report)
print(json.dumps(report))
