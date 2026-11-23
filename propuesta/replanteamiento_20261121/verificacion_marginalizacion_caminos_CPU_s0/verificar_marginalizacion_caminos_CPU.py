"""Independent enumeration and gradient checks; synthetic CPU only, no Porto labels."""
from base_replanteo import *
import torch
from marginalizacion_caminos_log import walk_duration_partition,ordinal_crps
start=time.time();torch.set_num_threads(2)
run=OUT/'verificacion_marginalizacion_caminos_CPU_s0';assert not run.exists();run.mkdir()
dtype=torch.float64
src=torch.tensor([0,0,0,1,1,2,2,3,3,4],dtype=torch.long)
dst=torch.tensor([0,1,2,1,3,0,3,1,3,5],dtype=torch.long)
weights=np.array([.2,.5,.3,.25,.6,.35,.65,.1,.3,.7],float)
a=torch.full((1,6),-torch.inf,dtype=dtype);a[0,0]=0
b=torch.full_like(a,-torch.inf);b[0,3]=0
lw=torch.tensor(np.log(weights)[None],dtype=dtype)
out=walk_duration_partition(a,b,src,dst,lw,2,8)
assert out['reachable'].item()
# Explicit path enumeration, not matrix recursion. Counts include repeated destination visits.
totals=np.zeros(9);paths=np.zeros(9,dtype=int);revisits=0
edges={i:[] for i in range(6)}
for u,v,w in zip(src.tolist(),dst.tolist(),weights):edges[u].append((v,float(w)))
def enumerate_paths(node,depth,mass,visited_destination):
 global revisits
 if node==3 and depth>=2:
  totals[depth]+=mass;paths[depth]+=1
  if visited_destination:revisits+=1
 if depth<8:
  for to,weight in edges[node]:enumerate_paths(to,depth+1,mass*weight,visited_destination or node==3)
enumerate_paths(0,0,1.,False)
expected=totals[2:]/totals[2:].sum()
max_error=float(np.max(np.abs(out['log_prob'].exp().numpy()[0]-expected)))
assert max_error<1e-13 and revisits>0 and abs(float(out['log_prob'].exp().sum())-1)<1e-13
# A separately implemented ordinary dense matrix multiplication is safe for this small fixture.
K=np.zeros((6,6))
for u,v,w in zip(src.tolist(),dst.tolist(),weights):K[u,v]+=w
alpha=np.eye(6)[0];dense=[]
for n in range(9):
 if n>=2:dense.append(alpha[3])
 alpha=alpha@K
assert np.max(np.abs(np.array(dense)-totals[2:]))<1e-13
# Differentiate a proper ordinal score plus categorical NLL through the global partition.
design=torch.tensor(np.random.default_rng(20261129).normal(size=(len(src),4)),dtype=dtype)
theta=torch.tensor([.2,-.1,.05,.3],dtype=dtype,requires_grad=True)
def loss_at(t):
 z=walk_duration_partition(a,b,src,dst,lw+(design@t)[None],2,8)
 return ordinal_crps(z['log_prob'],torch.tensor([3]))[0]-.02*z['log_prob'][0,3]
loss=loss_at(theta);loss.backward();analytic=theta.grad.detach().numpy();numeric=[]
eps=1e-5
for j in range(4):
 delta=torch.zeros(4,dtype=dtype);delta[j]=eps
 numeric.append(float((loss_at(theta.detach()+delta)-loss_at(theta.detach()-delta))/(2*eps)))
grad_error=float(np.max(np.abs(analytic-np.array(numeric))))
assert grad_error<1e-8 and np.isfinite(analytic).all()
# A genuinely disconnected terminal is exposed without zero-row deletion or NaN gradients.
unreachable_b=torch.full_like(b,-torch.inf);unreachable_b[0,5]=0
missing=walk_duration_partition(a,unreachable_b,src,dst,lw,2,8)
assert not missing['reachable'].item() and torch.isneginf(missing['log_partition']).item()
# Extreme path weights underflow in ordinary arithmetic, but the conditional mass stays normalized.
one=torch.zeros((1,1),dtype=dtype);ix=torch.zeros(1,dtype=torch.long)
deep=walk_duration_partition(one,one,ix,ix,torch.full((1,1),-1000.,dtype=dtype),4,119)
assert np.exp(-4000.)==0 and deep['reachable'].item()
assert torch.isfinite(deep['log_prob']).all() and abs(float(deep['log_prob'].exp().sum())-1)<1e-13
# Node permutation is a graph relabeling, not a change of query information.
perm=torch.tensor([3,0,5,2,1,4]);inverse=torch.argsort(perm)
relabeled=walk_duration_partition(a[:,perm],b[:,perm],inverse[src],inverse[dst],lw,2,8)
assert torch.allclose(relabeled['log_prob'],out['log_prob'],atol=1e-13,rtol=0)
# Batched independent queries, reverse and singleton.
aa=a.repeat(3,1);bb=b.repeat(3,1);ww=lw.repeat(3,1)+torch.tensor([0.,-.1,.15])[:,None]
batch=walk_duration_partition(aa,bb,src,dst,ww,2,8)
reverse=walk_duration_partition(aa.flip(0),bb.flip(0),src,dst,ww.flip(0),2,8)
singleton=walk_duration_partition(aa[:1],bb[:1],src,dst,ww[:1],2,8)
assert torch.equal(batch['log_prob'],reverse['log_prob'].flip(0))
assert torch.equal(batch['log_prob'][:1],singleton['log_prob'])
result=dict(passed=True,scope='Synthetic numerical instrument only; no trainedPorto model or timeMAE.',
 independent_exhaustive_paths=int(paths.sum()),repeated_destination_paths=revisits,
 max_probability_error=max_error,dense_weight_sum_exact=True,gradient_max_error=grad_error,
 unreachable_terminal_explicit=True,extreme_underflow_conditional_normalization=True,
 graph_relabeling_invariant=True,batch_reverse_singleton_exact=True,
 original_duration_support_s=[60,1785],no_querytrueN_or_route_used=True,
 model_claim=False,Validation=False,Test=False,seconds=time.time()-start,
 sources={name:sha(OUT/name) for name in ['verificar_marginalizacion_caminos_CPU.py','marginalizacion_caminos_log.py']})
write(run/'resultado.json',result)
for name in result['sources']:(run/name).write_bytes((OUT/name).read_bytes())
print(json.dumps(result))
