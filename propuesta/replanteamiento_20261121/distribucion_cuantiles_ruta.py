"""Marginalize latentroute conditionalquantiles without observedroute inputs."""
import numpy as np
from numba import njit,prange
@njit(cache=True,parallel=True)
def mixture_quantiles(conditional,weights):
 n,k,q=conditional.shape;out=np.empty((n,3),np.float64);targets=np.array([.1,.5,.9])
 for i in prange(n):
  for target_index in range(3):
   low=60.;high=1785.;target=targets[target_index]
   for iteration in range(36):
    mid=(low+high)*.5;cdf=0.
    for c in range(k):
     values=conditional[i,c];j=np.searchsorted(values,mid,side='right')
     if j==0:
      x0=60.;x1=values[0];p0=0.;p1=.1
     elif j==q:
      x0=values[q-1];x1=1785.;p0=.9;p1=1.
     else:
      x0=values[j-1];x1=values[j];p0=j*.1;p1=(j+1)*.1
     if x1<=x0:prob=p1
     else:prob=p0+(p1-p0)*min(max((mid-x0)/(x1-x0),0.),1.)
     cdf+=weights[i,c]*prob
    if cdf>=target:high=mid
    else:low=mid
   out[i,target_index]=high
 return out
def rearrange(quantiles):
 values=np.asarray(quantiles,np.float64);assert values.ndim==2 and values.shape[1]==9 and np.isfinite(values).all()
 return np.sort(np.clip(values,60,1785),axis=1)
