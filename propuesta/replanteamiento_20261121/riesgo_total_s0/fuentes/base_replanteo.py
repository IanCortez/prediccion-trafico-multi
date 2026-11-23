"""Shared isolated execution context; only local Conda and project files."""
from pathlib import Path
import sys,os,json,hashlib,time,types
import numpy as np
ROOT=Path('E:/UTEC/Tesis/TesisIan');BASE=ROOT/'propuesta';DATA=BASE/'datos';OUT=BASE/'replanteamiento_20261121';BEST=BASE/'experimentos/od_mezcla8_geometria_conjunta_s0_fp32_v2';FROZEN=BEST/'fuentes'
assert Path(sys.executable).resolve().is_relative_to(ROOT/'.conda-env')
sys.path.insert(0,str(FROZEN));common=types.ModuleType('common');common.__file__=str(BASE/'common.py');sys.modules['common']=common;exec(compile((FROZEN/'common.py').read_text(encoding='utf-8-sig'),common.__file__,'exec'),common.__dict__)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,o):
 Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else str(v)),encoding='utf-8')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def metric(y,p):
 y=np.asarray(y,np.float64);p=np.asarray(p,np.float64);assert y.shape==p.shape and y.ndim==1 and np.isfinite(p).all();e=np.abs(y-p)
 return dict(n=len(y),mae_s=float(e.mean()),rmse_s=float(np.sqrt(np.mean((y-p)**2))),mdae_s=float(np.median(e)))
def paired(y,a,b,t):
 days=np.asarray(t)//86400;_,i=np.unique(days,return_inverse=True);s=np.bincount(i,weights=np.abs(y.astype(float)-a)-np.abs(y.astype(float)-b));n=np.bincount(i);rg=np.random.default_rng(1729);ix=rg.integers(len(n),size=(3000,len(n)))
 return dict(delta_mae_s=float(s.sum()/n.sum()),ci95_s=np.quantile(s[ix].sum(1)/n[ix].sum(1),[.025,.975]),days=len(n),interpretation='Exploratory; reused Validation and adaptive model search')
def deny_test():
 accessed=set()
 def hook(event,args):
  if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
   try:p=Path(os.fsdecode(args[0])).resolve()
   except (ValueError,OSError):return
   if p.is_relative_to(DATA):
    assert not p.name.startswith('test'),'Test is sealed'
    accessed.add(str(p.relative_to(ROOT)))
 sys.addaudithook(hook);return accessed
