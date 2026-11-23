"""Dense covariance and independent scipy conditional-t oracle; no Porto data/GPU."""
from base_replanteo import *
import torch
from scipy.stats import multivariate_t,t as student_t
from puente_temporal_estudiante import *
start=time.time();torch.set_num_threads(2)
run=OUT/'verificacion_puente_temporal_CPU_s0';assert not run.exists();run.mkdir()
dtype=torch.float64;cov_errors=[]
for correlation in [0.,.5,.95,.999]:
 for n in [4,7,119]:
  i=np.arange(n);corr=correlation**np.abs(i[:,None]-i[None,:])
  a=accumulated_ar_scale(torch.tensor([[correlation]],dtype=dtype),119)[0,0,n-1].item()
  cov_errors.append(abs(a-corr.sum()))
assert max(cov_errors)<1e-8
n=7;mu=.13;scale=.045;rho=.73;nu=5.
increments=np.array([.07,.09,.22,.11,.16,.09,.26]);assert abs(increments.sum()-1)<1e-14
progress=np.cumsum(increments);index=np.arange(n)
cv=scale**2*rho**np.abs(index[:,None]-index[None,:])
A=np.tril(np.ones((n,n)));C=A@cv@A.T;mean=mu*np.arange(1,n+1)
ref_full=float(multivariate_t.logpdf(progress,loc=mean,shape=C,df=nu))
ref_end=float(student_t.logpdf(1,df=nu,loc=mean[-1],scale=np.sqrt(C[-1,-1])))
m=torch.tensor([[mu]],dtype=dtype);s=torch.tensor([[scale]],dtype=dtype);r=torch.tensor([[rho]],dtype=dtype)
v=torch.tensor(increments[None],dtype=dtype);target=torch.tensor([n])
full=log_increment_path(v,target,m,s,r,nu).item()
endpoint=log_endpoint_by_duration(m,s,r,nu)[0,0,n-4].item()
assert abs(full-ref_full)<1e-10 and abs(endpoint-ref_end)<1e-10,(full,ref_full,endpoint,ref_end)
# Independent conditional-t formula for interiorpositions given finalendpoint.
delta=1-mean[-1];cross=C[:-1,-1]
cond_mean=mean[:-1]+cross/C[-1,-1]*delta
cond_scale=(nu+delta**2/C[-1,-1])/(nu+1)*(C[:-1,:-1]-np.outer(cross,cross)/C[-1,-1])
ref_cond=float(multivariate_t.logpdf(progress[:-1],loc=cond_mean,shape=cond_scale,df=nu+1))
conditional=log_conditional_fit_path(v,target,m,s,r,torch.zeros_like(m),nu).item()
assert abs(conditional-ref_cond)<1e-10
# Differentiating actual properduration score plus normalized conditionalfitpath score.
theta=torch.tensor([-.2,-3.,.7],dtype=dtype,requires_grad=True)
def objective(parameters):
 mu=.02+parameters[0].sigmoid()[None,None]*.04
 sigma=parameters[1].exp()[None,None]
 correlation=parameters[2].sigmoid()[None,None]*.999
 prior=torch.zeros_like(mu)
 p=log_duration_posterior(mu,sigma,correlation,prior)
 return duration_score(p,target)[0]-.02*log_conditional_fit_path(v,target,mu,sigma,correlation,prior)[0]/(n-1)
loss=objective(theta);loss.backward();actual=theta.grad.numpy();numeric=[]
for j in range(3):
 h=torch.zeros(3,dtype=dtype);h[j]=1e-5
 numeric.append(float((objective(theta.detach()+h)-objective(theta.detach()-h))/(2e-5)))
grad_error=float(np.max(np.abs(actual-np.array(numeric))))
assert grad_error<1e-7 and np.isfinite(actual).all(),(grad_error,actual,numeric)
# Mixture normalization and parameter-only batched prediction, with no targetn argument.
rg=np.random.default_rng(20261130)
mu=torch.tensor(rg.uniform(.01,.1,size=(17,8)),dtype=dtype)
sigma=torch.tensor(rg.uniform(.001,.02,size=(17,8)),dtype=dtype)
rho=torch.tensor(rg.uniform(0,.999,size=(17,8)),dtype=dtype)
prior=torch.tensor(rg.normal(size=(17,8)),dtype=dtype)
p=log_duration_posterior(mu,sigma,rho,prior)
rev=log_duration_posterior(mu.flip(0),sigma.flip(0),rho.flip(0),prior.flip(0))
single=log_duration_posterior(mu[:1],sigma[:1],rho[:1],prior[:1])
assert torch.isfinite(p).all() and torch.max(torch.abs(p.exp().sum(1)-1))<1e-12
assert torch.equal(p,rev.flip(0)) and torch.equal(p[:1],single)
# Global-scale heavy tails reduce the influence of a large conflicting increment pair.
influence=[]
for amplitude in [10.,10000.]:
 noisy=v.clone();noisy[0,1]+=amplitude;noisy[0,3]-=amplitude;noisy.requires_grad_()
 score=-log_increment_path(noisy,target,m,s,r,nu).sum()
 score.backward();influence.append(abs(float(noisy.grad[0,1]-noisy.grad[0,3])))
assert influence[1]<influence[0]*.01,influence
proof=dict(passed=True,scope='Synthetic exactnumerical instrument; not a trainedPorto model or observedMAE.',
 stationary_AR_covariance_maxerror=max(cov_errors),full_t_logpdf_error=abs(full-ref_full),
 endpoint_t_logpdf_error=abs(endpoint-ref_end),conditional_t_logpdf_error=abs(conditional-ref_cond),
 gradient_maxerror=grad_error,normalized_fixed116durationposterior=True,
 batch_reverse_singleton_exact=True,no_querytrueN_in_prediction=True,
 robust_globalpathscale_influence=influence,robustness_is_not_independent_GPS_noise_model=True,
 firstpassage_claim=False,nonnegative_physicalspeed_claim=False,
 Validation=False,Test=False,seconds=time.time()-start,
 sources={name:sha(OUT/name) for name in ['puente_temporal_estudiante.py','verificar_puente_temporal_CPU.py']})
write(run/'resultado.json',proof)
for name in proof['sources']:(run/name).write_bytes((OUT/name).read_bytes())
print(json.dumps(proof))
