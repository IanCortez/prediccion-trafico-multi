"""Independent denseStudent-t/conditioning oracle and numericalgradient; syntheticCPUonly."""
from base_replanteo import *
import torch
from scipy.stats import multivariate_t
from puente_temporal_2d import *
start=time.time();torch.set_num_threads(2)
run=OUT/'verificacion_puente_2d_CPU_s0';assert not run.exists();run.mkdir()
dtype=torch.float64;errors=[]
for rho in [-.95,-.4,0.,.5,.95,.999]:
 for n in [4,7,119]:
  r=torch.tensor([[rho]],dtype=dtype);S,H=ar_sums(r);ix=np.arange(n)
  errors.append(abs(float(S[0,0,n-1])-(rho**np.abs(ix[:,None]-ix[None,:])).sum()))
  assert abs(float(H[0,0,n-1])-np.sum(rho**ix))<1e-12
assert max(errors)<1e-8
n=7;rho=.63;scale=np.array([.09,.07]);cross=.3;nu=5.
mu0=np.array([.09,.05]);muc=np.array([.12,0.])
increments=np.array([[.05,.02],[.1,.1],[.15,.15],[.16,-.02],[.18,-.07],[.17,-.08],[.19,-.1]])
assert np.allclose(increments.sum(0),[1,0])
ix=np.arange(n);Ctime=rho**np.abs(ix[:,None]-ix[None,:])
sigma=np.diag(scale)@np.array([[1,cross],[cross,1]])@np.diag(scale)
C=np.kron(Ctime,sigma)
mean=muc[None]+rho**ix[:,None]*(mu0-muc)[None]
ref_full=float(multivariate_t.logpdf(increments.ravel(),loc=mean.ravel(),shape=C,df=nu))
A=np.kron(np.tril(np.ones((n,n))),np.eye(2));positionC=A@C@A.T;positionmean=A@mean.ravel()
xy=np.cumsum(increments,axis=0).ravel();endxy=np.array([1.,0.])
ref_end=float(multivariate_t.logpdf(endxy,loc=positionmean[-2:],shape=positionC[-2:,-2:],df=nu))
p={key:torch.tensor(value,dtype=dtype) for key,value in dict(initial=mu0[None,None],cruise=muc[None,None],
 scale=scale[None,None],cross=[[cross]],rho=[[rho]],log_prior=[[0.]]).items()}
v=torch.tensor(increments[None],dtype=dtype);target=torch.tensor([n])
full=fullpath_logpdf(v,target,p).item();end=endpoint_logpdf(p)[0,0,n-4].item()
assert abs(full-ref_full)<1e-10 and abs(end-ref_end)<1e-10,(full,ref_full,end,ref_end)
# Independent conditionalStudent t on 2(n-1)interior positions, conditioned on final2Dpoint.
B=positionC[:-2,-2:];D=positionC[-2:,-2:];difference=endxy-positionmean[-2:]
conditionalmean=positionmean[:-2]+B@np.linalg.solve(D,difference)
conditionalscale=(nu+difference@np.linalg.solve(D,difference))/(nu+2)*(positionC[:-2,:-2]-B@np.linalg.solve(D,B.T))
ref_cond=float(multivariate_t.logpdf(xy[:-2],loc=conditionalmean,shape=conditionalscale,df=nu+2))
cond=conditional_fit_logpdf(v,target,p).item()
assert abs(cond-ref_cond)<1e-9,(cond,ref_cond)
# Padding is never a modelinput; its trainingtarget mask must remove it from the auxiliarylikelihood.
padded=torch.cat([v,torch.full((1,112,2),10000.,dtype=dtype)],1)
assert abs(fullpath_logpdf(padded,target,p).item()-full)<1e-12
theta=torch.tensor([-.2,-3.,.7,.2,-.1],dtype=dtype,requires_grad=True)
def make(t):
 meanC=torch.stack([.02+.03*t[0].sigmoid(),t[0]*.001])[None,None]
 mean0=meanC+torch.stack([.01*t[3].tanh(),.02*t[4].tanh()])[None,None]
 std=torch.stack([t[1].exp(),.8*t[1].exp()])[None,None]
 return dict(initial=mean0,cruise=meanC,scale=std,cross=.9*t[2].tanh()[None,None],
  rho=(-.95+1.945*t[4].sigmoid())[None,None],log_prior=torch.zeros((1,1),dtype=dtype))
def objective(t):
 par=make(t);logp,endpoint=duration_logp(par)
 return score(logp,target)[0][0]-.02*conditional_fit_logpdf(v,target,par,endpoint)[0]/(2*(n-1))
objective(theta).backward();analytic=theta.grad.numpy();numeric=[]
for j in range(len(theta)):
 delta=torch.zeros_like(theta);delta[j]=1e-5
 numeric.append(float((objective(theta.detach()+delta)-objective(theta.detach()-delta))/(2e-5)))
grad_error=float(np.max(np.abs(analytic-np.array(numeric))));assert grad_error<1e-7 and np.isfinite(analytic).all(),(grad_error,analytic,numeric)
rg=np.random.default_rng(20261130);batch=17;components=32
random_p=dict(initial=torch.tensor(rg.normal(.025,.02,size=(batch,components,2)),dtype=dtype),
 cruise=torch.tensor(rg.normal(.025,.01,size=(batch,components,2)),dtype=dtype),
 scale=torch.tensor(rg.uniform(.001,.02,size=(batch,components,2)),dtype=dtype),
 cross=torch.tensor(rg.uniform(-.9,.9,size=(batch,components)),dtype=dtype),
 rho=torch.tensor(rg.uniform(-.95,.995,size=(batch,components)),dtype=dtype),
 log_prior=torch.tensor(rg.normal(size=(batch,components)),dtype=dtype))
lp,_=duration_logp(random_p);reverse,_=duration_logp({k:x.flip(0) for k,x in random_p.items()});single,_=duration_logp({k:x[:1] for k,x in random_p.items()})
assert torch.isfinite(lp).all() and torch.max(torch.abs(lp.exp().sum(-1)-1))<1e-12
assert torch.equal(lp,reverse.flip(0)) and torch.equal(lp[:1],single)
qt=quantiles(lp);assert bool(((qt>=60)&(qt<=1785)&(qt%15==0)).all())
assert bool((qt[:,0]<=qt[:,1]).all() and (qt[:,1]<=qt[:,2]).all())
influence=[]
for amplitude in [10.,10000.]:
 noisy=v.clone();noisy[0,1,0]+=amplitude;noisy[0,3,0]-=amplitude;noisy.requires_grad_()
 (-fullpath_logpdf(noisy,target,p).sum()).backward()
 influence.append(abs(float(noisy.grad[0,1,0]-noisy.grad[0,3,0])))
assert influence[1]<influence[0]*.01
proof=dict(passed=True,scope='Synthetic2Drelaxingmean exactStudentbridge, no PortoMAE.',
 AR_sums_maxerror=max(errors),fullpath_logpdf_error=abs(full-ref_full),endpoint_logpdf_error=abs(end-ref_end),
 conditional_logpdf_error=abs(cond-ref_cond),gradient_maxerror=grad_error,
 training_padding_excluded=True,fixed116duration_normalization=True,reverse_singleton_exact=True,
 ordered_original15sec_quantiles=True,globalpathscale_influence=influence,
 no_true_queryN_or_GPS_in_prediction=True,Validation=False,Test=False,seconds=time.time()-start,
 sources={name:sha(OUT/name) for name in ['puente_temporal_2d.py','verificar_puente_2d_CPU.py']})
write(run/'resultado.json',proof)
for name in proof['sources']:(run/name).write_bytes((OUT/name).read_bytes())
print(json.dumps(proof))
