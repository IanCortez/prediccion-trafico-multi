from base_replanteo import *
from modelo_prefijo_nativo import NativePrefixArrival,counter_uniform
from datos_prefijo_nativo import NativeTrainSequences,query_seeds
from torch import nn
import torch
torch.set_num_threads(4)
torch.manual_seed(71)
access=deny_test()
offsets=np.array([(a,b) for a in range(-3,4) for b in range(-3,4) if (a,b)!=(0,0)])
neighbors=np.full((3,48),-1,np.int64)
for cell in range(3):
    for j,(dx,dy) in enumerate(offsets):
        if dy==0 and 0<=cell+dx<3:neighbors[cell,j]=cell+dx
allowed=neighbors>=0
graph=dict(neighbors=neighbors,allowed=allowed,centers=np.array([[0.,0.],[200.,0.],[400.,0.]],np.float32),offsets=offsets)
grid=dict(nx=3,ny=1,x0=0,y0=0,cell_m=200)
stats=dict(mean=np.zeros(8),std=np.ones(8),keep=np.arange(8),bins=[torch.tensor([-3.,0.,3.]) for _ in range(8)])
init=dict(lags=[0,0,1,1,2,2,3,3],weights=[.125]*8,hazards=[.5]*8)
m=NativePrefixArrival(stats,grid,graph,np.abs(np.arange(3)[:,None]-np.arange(3)[None,:])*200,[4.,.5],np.zeros((5,3*160,48)),init).eval()
with torch.no_grad():m.history_projection.weight.normal_(0,.03)
x=torch.tensor([[0.,0.,.4,0.,.4,0.,.4,-.9],[0.,0.,.4,0.,.4,0.,.4,-.9]])
o=torch.zeros(2,dtype=torch.long);d=torch.full_like(o,2)
cell=torch.tensor([[0,0,1,1,2,2],[0,1,1,2,2,2]])
head=torch.tensor([[4,4,0,0,0,0],[4,0,0,0,0,0]])
age=torch.tensor([[0,1,0,1,0,1],[0,0,1,0,1,2]])
motion,stop=m.teacher_probabilities(x,o,d,cell,head,age)
q,*_=m.encode_od(x,o,d)
hidden=q[None]
parts=[];ends=[]
for t in range(cell.shape[1]):
    a,b,hidden=m.rollout_step(q,x[:,:4],cell[:,t],d,head[:,t],age[:,t],t,hidden)
    parts.append(a);ends.append(b)
error=float((torch.stack(parts,1)-motion).abs().max().detach())
assert error<1e-6 and torch.allclose(torch.stack(ends,1),stop,atol=1e-6,rtol=0)
changed=cell.clone();changed[:,3:]=2-changed[:,3:]
a,b=m.teacher_probabilities(x,o,d,changed,head,age)
assert torch.allclose(a[:,:3],motion[:,:3],atol=1e-7,rtol=0)
assert torch.allclose(b[:,:3],stop[:,:3],atol=1e-7,rtol=0)
short_a,short_b=m.teacher_probabilities(x,o,d,cell[:,:3],head[:,:3],age[:,:3])
assert torch.allclose(short_a,motion[:,:3],atol=1e-6,rtol=0)
move_index=int(np.flatnonzero((offsets==[1,0]).all(-1))[0])+1
loss=-motion[:,0,move_index].log().mean()-stop[:,-1].clamp_min(1e-9).log().mean()
loss.backward()
assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
assert m.prefix_rnn.weight_ih_l0.grad.abs().sum()>0
class ExactChain(NativePrefixArrival):
    def __init__(self):
        nn.Module.__init__(self)
        self.register_buffer('neighbors',torch.ones(2,48,dtype=torch.long))
        self.register_buffer('outgoing_heading',torch.zeros(48,dtype=torch.long))
    def encode_od(self,x,origin,destination):return (torch.zeros(len(x),64),)
    def rollout_step(self,q,geo,cell,destination,heading,age,elapsed,hidden):
        motion=torch.zeros(len(cell),49)
        motion[:,0]=(cell==1).float()
        motion[:,1]=(cell==0).float()
        return motion,.25*(cell==1).float(),hidden
f=ExactChain().eval()
xx=torch.zeros(3,8);oo=torch.zeros(3,dtype=torch.long);dd=torch.ones_like(oo);seeds=torch.tensor([31,19,42])
raw,survival=f.generated_arrival(xx,oo,dd,seeds,particles=17,horizon=20)
expected=torch.zeros(20);expected[1:]=.25*.75**torch.arange(19)
assert torch.allclose(raw,expected[None].expand(3,-1),atol=1e-7,rtol=0)
assert torch.allclose(raw.sum(-1)+survival,torch.ones(3),atol=1e-6,rtol=0)
order=torch.tensor([2,0,1])
rr,_=f.generated_arrival(xx[order],oo[order],dd[order],seeds[order],particles=17,horizon=20)
assert torch.equal(rr,raw[order])
for i in range(3):
    one,_=f.generated_arrival(xx[i:i+1],oo[i:i+1],dd[i:i+1],seeds[i:i+1],particles=17,horizon=20)
    assert torch.equal(one,raw[i:i+1])
uniform=counter_uniform(torch.full((10000,),29),torch.arange(10000),11)
assert torch.all((uniform>0)&(uniform<1)) and abs(float(uniform.mean())-.5)<.015
tr=dict(np.load(DATA/'train_x.npz'));y=np.load(DATA/'train_y.npy')
g=read(OUT/'datos_semi_markov/protocolo.json')
realgraph=np.load(OUT/'datos_semi_markov/grafo_grid.npz')
N=len(realgraph['centers'])
def ids(tag):
    xx=(tr[tag+'_lon']+8.62)*111195*np.cos(np.deg2rad(41.16))
    yy=(tr[tag+'_lat']-41.16)*111195
    return np.floor((xx-g['x0'])/200).astype(np.int64)*g['ny']+np.floor((yy-g['y0'])/200).astype(np.int64)
origin,destination=ids('o'),ids('d')
seq=NativeTrainSequences(OUT/'datos_semi_markov/runs_train.npz',realgraph['neighbors'],realgraph['allowed'],g['ny'],N)
ii=np.random.default_rng(71).choice(len(seq.trip),1000,replace=False)
rows,batch=seq.make(ii,origin,destination,y)
assert batch['terminal'].sum()==1000
points=int(batch['event_mask'].sum())
motion_valid=int(batch['motion_mask'].sum())
sd={k:tr[k][rows[:3]] for k in ['o_lon','o_lat','d_lon','d_lat','timestamp']}
s0=query_seeds(sd)
s1=query_seeds({k:np.repeat(v[:1],2) for k,v in sd.items()})
assert s1[0]==s1[1]
assert np.array_equal(query_seeds({k:v[::-1] for k,v in sd.items()}),s0[::-1])
assert not torch.cuda.is_initialized()
report=dict(passed=True,teacher_vs_incremental_max_error=error,future_intervention_causal=True,padding_length_does_not_enter_prefix=True,finite_gradients_and_nonzero_RNN_gradient=True,exact_geometric_absorption_checked=True,unconditioned_mass_and_survival_conserved=True,stateless_seed_permutation_and_singleton=True,real_Train_trips_reconstructed=1000,duration_labels_and_endpoints_exact=True,valid_point_states=points,valid_motion_targets=motion_valid,query_inputs='OD and departure; simulated history only; horizon120 fixed, not actual query N',no_GPU=True,Test_access=False,actual_files_accessed=sorted(access),code_sha256=sha(__file__),model_sha256=sha(OUT/'modelo_prefijo_nativo.py'),data_sha256=sha(OUT/'datos_prefijo_nativo.py'))
write(OUT/'verificacion_prefijo_nativo.json',report)
print(json.dumps(report))
