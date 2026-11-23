"""Deterministic empirical leaf duration distributions and explicit supervised duration regimes."""
from base_replanteo import *
from sklearn.ensemble import ExtraTreesRegressor,ExtraTreesClassifier
GRID=np.arange(60,1786,15,dtype=np.float64)
def quantiles(prob):
 cdf=prob.cumsum(1)
 return np.column_stack([GRID[np.minimum((cdf<q).sum(1),115)] for q in [.1,.5,.9]])
class LeafDurationForest:
 def __init__(self,trees=256,seed=20261122):
  self.forest=ExtraTreesRegressor(n_estimators=trees,max_depth=14,min_samples_leaf=20,max_features=1.,
    n_jobs=2,random_state=seed,criterion='squared_error')
 def fit(self,X,y):
  y=np.asarray(y);label=((y-60)/15).astype(int)
  assert np.array_equal(GRID[label],y)
  self.forest.fit(X,y);self.lookup=[];self.banks=[]
  prior=np.bincount(label,minlength=116).astype(float);prior/=prior.sum()
  for estimator in self.forest.estimators_:
   leaves=estimator.apply(X);unique,index=np.unique(leaves,return_inverse=True)
   counts=np.bincount(index*116+label,minlength=len(unique)*116).reshape(-1,116).astype(float)
   # One marginal pseudo-observation for every leaf; fixed before held labels.
   prob=(counts+prior[None])/(counts.sum(1,keepdims=True)+1)
   lookup=np.full(estimator.tree_.node_count,-1,np.int32);lookup[unique]=np.arange(len(unique))
   self.lookup.append(lookup);self.banks.append(prob.astype(np.float32))
  return self
 def predict_proba(self,X):
  parts=[]
  for first in range(0,len(X),512):
   take=X[first:first+512];prob=np.zeros((len(take),116),np.float64)
   for tree,lookup,bank in zip(self.forest.estimators_,self.lookup,self.banks):prob+=bank[lookup[tree.apply(take)]]
   prob/=len(self.banks);prob/=prob.sum(1,keepdims=True);parts.append(prob)
  return np.concatenate(parts)
 def predict_quantiles(self,X):return quantiles(self.predict_proba(X))
class DurationRegimeForest:
 def __init__(self,seed=20261122):
  self.gate=ExtraTreesClassifier(n_estimators=128,max_depth=14,min_samples_leaf=20,max_features=1.,n_jobs=2,random_state=seed,criterion='log_loss')
  self.experts=[LeafDurationForest(128,seed),LeafDurationForest(128,seed)]
 def fit(self,X,y):
  regime=(y>=900).astype(int);self.gate.fit(X,regime)
  for k in range(2):self.experts[k].fit(X[regime==k],y[regime==k])
  return self
 def predict_proba(self,X):
  gate=self.gate.predict_proba(X);a=self.experts[0].predict_proba(X);b=self.experts[1].predict_proba(X)
  assert np.all(a[:,56:]==0) and np.all(b[:,:56]==0)
  out=gate[:,:1]*a+gate[:,1:]*b
  out/=out.sum(1,keepdims=True);return out,gate[:,1]
 def predict_quantiles(self,X):
  out,gate=self.predict_proba(X);return quantiles(out),gate
