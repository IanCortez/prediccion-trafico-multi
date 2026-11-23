"""Fixed historical source density and evidence gate; no target-dependent selection."""
from base_replanteo import *
from scipy.special import ndtr
GRID=np.arange(60,1786,15,dtype=np.float64)
EDGES=np.arange(67.5,1785,15,dtype=np.float64)
def source_distribution(source):
 parent,first,last,duration,logweight,error,full=source
 valid=parent>=0
 weights=np.exp(logweight-logweight.max(1,keepdims=True))*valid
 weights/=np.maximum(weights.sum(1,keepdims=True),1e-30)
 ess=1/np.maximum((weights**2).sum(1),1e-30)
 ess=np.where(valid.any(1),ess,0)
 safe_duration=np.maximum(duration,60)
 # Fixed 20% multiplicative source uncertainty, censored onto original support.
 z=(np.log(EDGES)[None,None,:]-np.log(safe_duration)[:,:,None])/.20
 each=ndtr(z)
 cdf=(weights[:,:,None]*each).sum(1)
 prob=np.diff(np.concatenate([np.zeros((len(parent),1)),cdf,np.ones((len(parent),1))],1),axis=1)
 error2=np.sum(weights*np.where(valid,error,0)**2,axis=1)
 gate=ess/(ess+8)*np.exp(-.5*error2/(50*50))
 assert np.isfinite(prob).all() and np.min(prob)>-1e-10
 prob=np.maximum(prob,0);prob/=prob.sum(1,keepdims=True)
 summary=np.column_stack([valid.sum(1),ess,np.sqrt(error2),gate,(weights*full).sum(1),
                          (weights*duration).sum(1)])
 return prob,gate,summary
def quantiles(prob):
 cdf=prob.cumsum(1)
 return np.column_stack([GRID[np.minimum((cdf<q).sum(1),115)] for q in [.1,.5,.9]])
def combine(base,source):
 prob,gate,summary=source_distribution(source)
 exists=summary[:,0]>0
 direct=np.where(exists[:,None],prob,base)
 mixture=(1-gate[:,None])*base+gate[:,None]*prob
 return dict(kernel_backoff=quantiles(direct),evidence_mixture=quantiles(mixture)),summary
