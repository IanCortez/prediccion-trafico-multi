from base_replanteo import *
from historia_eventos import TemporalEventHistory,event_groups
import torch
rng=np.random.default_rng(67)
h=rng.poisson(.2,size=(5,120,120,2)).astype(np.uint16)
h[:,np.arange(120)//12%2==0,:,1]=0
original=TemporalEventHistory(h)
group=np.array([0,3,12,37,81,119])[:,None]
age=np.array([0,1,4,19,119])[None,:]
p=original.lookup(group,age,0)
changed=h.copy()
changed[0]+=37
mutated=TemporalEventHistory(changed)
q=mutated.lookup(group,age,0)
zero=h.copy();zero[0]=0
only_fit=TemporalEventHistory(zero).lookup(group,age)
assert np.array_equal(p,q) and np.array_equal(p,only_fit)
assert p.shape==(6,5,3) and np.all(p>=0) and np.allclose(p.sum(-1),1,atol=2e-7)
cell=np.full(15,13);heading=np.zeros(15,dtype=np.int64)
destination=np.full(15,13);t=np.arange(40,55);a=np.arange(15)
g=event_groups(cell,heading,destination,1460041000,t,a)
assert np.all(g==g[0])
rev=np.arange(14,-1,-1)
assert np.array_equal(event_groups(cell[rev],heading[rev],destination[rev],1460041000,t[rev],a[rev]),g[rev])
assert not torch.cuda.is_initialized()
report=dict(passed=True,entire_fold_intervention_invariant_at_every_backoff=True,explicit_removal_of_held_fold_exact=True,broadcast_shapes_and_probabilities=True,run_entry_clock_constant_through_a_run=True,query_order_invariant=True,no_query_duration_argument=True,no_GPU=True,code_sha256=sha(__file__),history_code_sha256=sha(OUT/'historia_eventos.py'))
write(OUT/'verificacion_historia_eventos.json',report)
print(json.dumps(report))
