from base_replanteo import *
from modelo_dir_prior import PriorDirectionalArrival
import torch,math
torch.set_num_threads(4);torch.manual_seed(0);access=deny_test();ck=torch.load(BEST/'modelo.pt',map_location='cpu',weights_only=False);grid=read(OUT/'datos_semi_markov/protocolo.json');graph=dict(np.load(OUT/'datos_semi_markov/grafo_grid.npz'));distance=np.load(OUT/'datos_semi_markov/distancias_grid.npy');counts=np.load(OUT/'prior_transiciones/conteos_fold.npz')['counts'];m=PriorDirectionalArrival(ck,grid,graph,distance,ck['anchors'][3],counts).eval();rg=np.random.default_rng(2);c=torch.tensor(rg.integers(m.N,size=64));h=torch.tensor(rg.integers(5,size=64));geo=torch.tensor(rg.normal(size=(64,4)),dtype=torch.float32);f=torch.tensor(rg.integers(5,size=64));rel=(geo[:,2:4]*1000-m.centers[c]).numpy();angle=np.floor((np.arctan2(rel[:,1],rel[:,0])+np.pi/8)/(np.pi/4)).astype(int)%8;db=np.searchsorted([500,1500,4000],np.linalg.norm(rel,axis=1));co=angle*4+db;g=c.numpy()*32+co;hg=g*5+h.numpy();ref=[]
for i in range(64):
 sub=counts[np.arange(5)!=int(f[i])].sum(0,dtype=np.uint64);ch=sub[hg[i]].astype(float);cb=sub.reshape(m.N*32,5,48)[g[i]].sum(0,dtype=np.uint64).astype(float);cg=sub.reshape(m.N,32,5,48)[:,co[i],:,:].sum((0,1),dtype=np.uint64).astype(float);gp=(cg+1)/(cg.sum()+48);p0=(cb+20*gp)/(cb.sum()+20);ref.append((ch+20*p0)/(ch.sum()+20))
with torch.no_grad():p=m.prior_probabilities(geo,c,h,f)
error=float((p-torch.tensor(np.array(ref))).abs().max());assert error<2e-6 and (p.sum(-1)-1).abs().max()<2e-6
# Counterfactual intervention adds counts ONLY to each lookup aggregate of fold0.
zero=torch.zeros(64,dtype=torch.long)
with torch.no_grad():
 before=m.prior_probabilities(geo,c,h,zero).clone();full_before=m.prior_probabilities(geo,c,h).clone();b=int(g[0]);hh=int(hg[0]);cc=int(co[0]);action=0;delta=100.
 changes=[(m.counts_total,(hh,action)),(m.counts_fold,(0,hh,action)),(m.counts_base,(b,action)),(m.counts_base_fold,(0,b,action)),(m.counts_global,(cc,action)),(m.counts_global_fold,(0,cc,action))]
 for t,index in changes:t[index]+=delta
 after=m.prior_probabilities(geo,c,h,zero);full_after=m.prior_probabilities(geo,c,h);assert torch.equal(before,after);assert (full_after-full_before).abs().max()>0
 for t,index in changes:t[index]-=delta
 q=torch.randn(64,64);dest=torch.tensor(rg.integers(m.N,size=64));pp,ww=m.state_parameters(q,geo,c,dest,h,f);assert torch.isfinite(pp).all() and torch.isfinite(ww).all();assert torch.all((pp.sum(-1)<1e-7)|((pp.sum(-1)-1).abs()<2e-6))
x=torch.zeros(1,216);x[:,ck['keep']]=torch.as_tensor(ck['mean']);x[0,:4]=torch.tensor([0.,0.,1.,1.]);x[0,6]=2**.5
with torch.no_grad():p,mass,p0,path=m.posterior(x,excluded_fold=torch.tensor([0]))
assert torch.isfinite(p).all() and abs(float(p.sum())-1)<1e-5 and 0<=float(mass)<=1.0001
report=dict(passed=True,NumPy_backoff_reference_max_error=error,own_fold_count_intervention_invariant=True,full_prior_reacts_to_added_count=True,local_policy_and_wait_finite=True,full_posterior_normalized=True,synthetic_absorption_mass=float(mass),query_GPS_or_duration_used=False,counts_sha256=sha(OUT/'prior_transiciones/conteos_fold.npz'),code_sha256=sha(__file__),model_code_sha256=sha(OUT/'modelo_dir_prior.py'),Test_access=False)
write(OUT/'verificacion_prior_cpu.json',report);print(json.dumps(report))

