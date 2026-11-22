"""Bounded regional transfer pilot. Only permitted target/source labels, no Test."""
from pathlib import Path
import sys, os, json, time, hashlib, types
import numpy as np
ROOT=Path('E:/UTEC/Tesis/TesisIan'); BASE=ROOT/'propuesta'; OUT=BASE/'desarrollo_20261121'
FROZEN=BASE/'experimentos/od_mezcla8_geometria_conjunta_s0_fp32_v2/fuentes'
assert Path(sys.executable).resolve().is_relative_to(ROOT/'.conda-env')
sys.path.insert(0,str(FROZEN))
common=types.ModuleType('common');common.__file__=str(BASE/'common.py');sys.modules['common']=common
exec(compile((FROZEN/'common.py').read_text(encoding='utf-8-sig'),common.__file__,'exec'),common.__dict__)
from inferencia import feature_input
from catboost import CatBoostRegressor
import pandas as pd

def write(p,o):
 p.write_text(json.dumps(o,ensure_ascii=False,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist() if isinstance(v,np.ndarray) else str(v)),encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def metric(y,p):
 assert len(y)==len(p) and np.isfinite(p).all()
 return dict(n=len(y),mae_s=float(np.abs(y-p).mean()),rmse_s=float(np.sqrt(np.mean((y-p)**2))))
def paired(y,a,b,t):
 d=np.asarray(t)//86400;_,i=np.unique(d,return_inverse=True);s=np.bincount(i,weights=np.abs(y-a)-np.abs(y-b));n=np.bincount(i);rg=np.random.default_rng(1729);ix=rg.integers(len(n),size=(3000,len(n)))
 return dict(delta_mae_s=float(s.sum()/n.sum()),ci95_s=np.quantile(s[ix].sum(1)/n[ix].sum(1),[.025,.975]),days=len(n))
def groups(d):
 h=np.asarray(pd.to_datetime(d['timestamp'],unit='s',utc=True).tz_convert('Europe/Lisbon').hour)
 return (d['od_m']>=2000).astype(int)*2+((h>=16)&(h<20)).astype(int)
def main():
 start=time.time();run=OUT/'piloto_transferencia';run.mkdir(exist_ok=False)
 accessed=[]
 def audit(event,args):
  if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
   try:p=Path(os.fsdecode(args[0])).resolve()
   except (ValueError,OSError):return
   if p.is_relative_to(BASE/'datos'):
    assert not p.name.startswith('test'), 'Test is sealed'
    if str(p) not in accessed:accessed.append(str(p))
 sys.addaudithook(audit)
 cfg=dict(routes=True,context=True,projected_routes=True,mapmatched=False,calendar=True)
 protocol=dict(target_rule='both endpoints longitude >= -8.62',source_rule='both endpoints longitude < -8.62',target_label_budget_total=5000,target_fit_n=4000,target_selection_n=1000,source_n=20000,seed=20261121,feature_config=cfg,maximum_fits=3,maximum_configurations=4,max_wall_seconds=7200,max_additional_bytes=1000000000,Test_access=False,goal_global_mae_s=73.5,scope='Regional transfer diagnostic; not global goal confirmation')
 write(run/'protocolo.json',protocol)
 tr,X,names=feature_input('train',dict(config=cfg,architecture='transfer_pilot'))
 va,V,vnames=feature_input('validation',dict(config=cfg,architecture='transfer_pilot'))
 assert names==vnames and len(names)==206 and not any(s.startswith('mm_') for s in names)
 y=np.load(BASE/'datos/train_y.npy');v=np.load(BASE/'datos/validation_y.npy')
 target=np.flatnonzero((tr['o_lon']>=-8.62)&(tr['d_lon']>=-8.62));source=np.flatnonzero((tr['o_lon']<-8.62)&(tr['d_lon']<-8.62));outer=np.flatnonzero((va['o_lon']>=-8.62)&(va['d_lon']>=-8.62))
 # Reserve the final chronological fifth of TRAIN for selecting the gate.
 cutoff=int(np.quantile(tr['timestamp'],.8));rg=np.random.default_rng(20261121)
 fit=np.sort(rg.choice(target[tr['timestamp'][target]<cutoff],4000,replace=False));sel=np.sort(rg.choice(target[tr['timestamp'][target]>=cutoff],1000,replace=False));src=np.sort(rg.choice(source[tr['timestamp'][source]<cutoff],20000,replace=False))
 assert not np.intersect1d(fit,sel).size and not np.intersect1d(np.r_[fit,sel],src).size
 assert tr['timestamp'][fit].max()<tr['timestamp'][sel].min() and tr['timestamp'][sel].max()<va['timestamp'].min()
 np.savez_compressed(run/'indices.npz',fit_row=tr['row'][fit],selection_row=tr['row'][sel],source_row=tr['row'][src],evaluation_row=va['row'][outer])
 params=dict(task_type='GPU',devices='0',loss_function='MAE',eval_metric='MAE',iterations=1200,depth=7,learning_rate=.05,l2_leaf_reg=10,random_seed=20261121,allow_writing_files=False,verbose=200,thread_count=4)
 base=CatBoostRegressor(**params);base.fit(X[fit],y[fit]/600,eval_set=(X[sel],y[sel]/600),early_stopping_rounds=120)
 bs=base.predict(X[sel])*600;bv=base.predict(V[outer])*600;bt=base.predict(X[fit])*600
 base.save_model(str(run/'base.cbm'))
 # Matched added capacity: same architecture/training budget in both corrections.
 p=params|dict(iterations=700,depth=6,random_seed=20261122)
 target_c=CatBoostRegressor(**p);target_c.fit(X[fit],(y[fit]-bt)/600,eval_set=(X[sel],(y[sel]-bs)/600),early_stopping_rounds=100)
 tv=np.clip(bv+target_c.predict(V[outer])*600,60,1785);target_c.save_model(str(run/'correccion_objetivo.cbm'))
 pool=np.r_[fit,src];res=(y[pool]-base.predict(X[pool])*600)/600;weights=np.r_[np.ones(len(fit)),np.full(len(src),len(fit)/len(src))]
 source_c=CatBoostRegressor(**p);source_c.fit(X[pool],res,sample_weight=weights,eval_set=(X[sel],(y[sel]-bs)/600),early_stopping_rounds=100)
 cs=source_c.predict(X[sel])*600;cv=source_c.predict(V[outer])*600;source_c.save_model(str(run/'correccion_fuentes.cbm'))
 gs=groups({k:a[sel] for k,a in tr.items()});gv=groups({k:a[outer] for k,a in va.items()});alpha=np.zeros(4);gate=[]
 for g in range(4):
  m=gs==g;choices=np.array([0,.25,.5,.75,1]);loss=np.array([np.abs(y[sel][m]-np.clip(bs[m]+a*cs[m],60,1785)).mean() for a in choices]);a=choices[loss.argmin()]
  # Conservative support and block comparison on target selection labels only.
  evidence=paired(y[sel][m],np.clip(bs[m]+a*cs[m],60,1785),np.clip(bs[m],60,1785),tr['timestamp'][sel][m])
  if m.sum()>=100 and evidence['ci95_s'][1]<0:alpha[g]=a
  gate.append(dict(group=g,n=int(m.sum()),candidate_alpha=float(a),accepted_alpha=float(alpha[g]),selection_mae_s=loss,evidence=evidence))
 preds={'objetivo_solo':np.clip(bv,60,1785),'capacidad_objetivo':tv,'fuentes_sin_seleccion':np.clip(bv+cv,60,1785),'fuentes_selectivas':np.clip(bv+alpha[gv]*cv,60,1785)}
 assert np.array_equal(preds['fuentes_selectivas'][alpha[gv]==0],preds['objetivo_solo'][alpha[gv]==0])
 controls=['objetivo_solo','capacidad_objetivo'];best=min(controls,key=lambda k:metric(v[outer],preds[k])['mae_s']);improvement=1-metric(v[outer],preds['fuentes_selectivas'])['mae_s']/metric(v[outer],preds[best])['mae_s'];delta=paired(v[outer],preds['fuentes_selectivas'],preds[best],va['timestamp'][outer]);passed=improvement>=.03 and delta['ci95_s'][1]<0
 np.savez_compressed(run/'predicciones.npz',row=va['row'][outer],timestamp=va['timestamp'][outer],true=v[outer],**preds)
 result=dict(protocol=protocol,features=names,metrics={k:metric(v[outer],p) for k,p in preds.items()},gate=gate,paired_vs_best_control=delta,relative_improvement=float(improvement),passed_expansion_gate=bool(passed),seconds=time.time()-start,data_files_accessed=accessed,code_sha256=sha(Path(__file__)),model_hashes={p.name:sha(p) for p in run.glob('*.cbm')},conclusion='Expandir solo si mejora >=3% al mejor control y CI pareado por dia excluye cero. Resultado regional, no confirmacion del objetivo global.')
 write(run/'resultado.json',result);print(json.dumps({'final':result['metrics'],'passed_expansion_gate':passed,'seconds':result['seconds']}),flush=True)
if __name__=='__main__':main()
