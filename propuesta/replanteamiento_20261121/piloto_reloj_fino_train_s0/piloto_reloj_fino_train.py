"""Train-only chronological gate for finer deterministic departure-time cycles."""
from base_replanteo import *
import shutil
from catboost import CatBoostRegressor
from inferencia import feature_input
access=deny_test();start=time.time();run=OUT/'piloto_reloj_fino_train_s0'
assert not run.exists();run.mkdir();shutil.copy2(__file__,run/Path(__file__).name)
plan=dict(scope='Train-only chronological gate, not primary Validation or confirmation',
 seed=20261122,fit_n=200000,held_n=50000,held_last_days=21,
 embargo='Every fit trip must ARRIVE before first held departure; eligible primary Train only.',
 periods_seconds=[15,30,45,60,90,120,180,240,300],
 query=['o_lon','o_lat','d_lon','d_lat','timestamp'],
 hypothesis='Finite subminute/multiminute periodic functions may expose departure-time structure absent from coarse calendar encodings. This does not assume observed signal states.',
 public_features='Original216 minus every mm_ historical movement feature; static public features only. No fitted pretrained representation.',
 model=dict(iterations=800,depth=7,learning_rate=.05,l2_leaf_reg=5,loss_function='MAE',thread_count=2,random_seed=20261122,allow_writing_files=False,task_type='CPU'),
 checkpoint='Fixed final800 trees for both; no held stopping or parameter search.',
 gate='At least0.5second heldTrain MAE improvement AND upper95%day-bootstrap delta interval below0.',
 GPU=False,Validation=False,Test=False,preserve='All metrics,held predictions,sources,parameters; small final models retained for inspection.')
write(run/'plan.json',plan)
rec=read(BEST/'resultado.json');d,X,names=feature_input('train',rec)
keep=np.array([not n.startswith('mm_') for n in names]);assert int(keep.sum())==206
public_names=[n for n,k in zip(names,keep) if k]
assert not any('call_' in n or 'driver' in n for n in public_names)
y=np.load(DATA/'train_y.npy');eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index']
assert len(eligible)==1023485 and len(np.unique(eligible))==len(eligible)
ts=np.asarray(d['timestamp'],dtype=np.int64);split=int(ts[eligible].max())-21*86400
fit_pool=eligible[ts[eligible]+y[eligible]<split];held_pool=eligible[ts[eligible]>=split]
assert len(fit_pool)>=plan['fit_n'] and len(held_pool)>=plan['held_n']
rng=np.random.default_rng(plan['seed']);fit=np.sort(rng.choice(fit_pool,plan['fit_n'],replace=False));held=np.sort(rng.choice(held_pool,plan['held_n'],replace=False))
assert np.intersect1d(fit,held).size==0 and (ts[fit]+y[fit]).max()<ts[held].min()
ix=np.concatenate([fit,held]);base=X[ix][:,keep].copy();del X
def cycles(t):
 out=[]
 for period in plan['periods_seconds']:
  phase=2*np.pi*(np.asarray(t,np.int64)%period)/period
  out.extend([np.sin(phase),np.cos(phase)])
 return np.stack(out,axis=1).astype(np.float32)
fine=cycles(ts[ix]);assert np.array_equal(fine,cycles(ts[ix]+3600))
assert np.isfinite(base).all() and np.isfinite(fine).all()
n=len(fit);predictions={};metrics={};models={}
for mode in ['control','cyclic']:
 data=base if mode=='control' else np.concatenate([base,fine],axis=1)
 model=CatBoostRegressor(**plan['model'])
 model.fit(data[:n],y[fit],verbose=False)
 prediction=model.predict(data[n:]);predictions[mode]=prediction
 metrics[mode]=metric(y[held],prediction)
 model.save_model(str(run/(mode+'.cbm')));models[mode]=sha(run/(mode+'.cbm'))
 print(json.dumps(dict(stage=mode,validation_within_Train=metrics[mode],seconds=time.time()-start)),flush=True)
comparison=paired(y[held],predictions['cyclic'],predictions['control'],ts[held])
comparison['interpretation']='Exploratory chronological heldTrain comparison; no Validation labels or model-selected stopping.'
passed=bool(comparison['delta_mae_s']<=-.5 and comparison['ci95_s'][1]<0)
np.savez_compressed(run/'predicciones_held_Train.npz',row=d['row'][held],timestamp=ts[held],true=y[held],**predictions)
write(run/'resultado.json',dict(plan=plan,passed=passed,metrics=metrics,paired=comparison,
 split_timestamp=split,fit_pool_n=len(fit_pool),held_pool_n=len(held_pool),fit_last_arrival=int((ts[fit]+y[fit]).max()),held_first_departure=int(ts[held].min()),
 public_feature_names=public_names,cyclic_features=18,one_hour_periodicity_exact=True,
 selected_fit_rows_sha256=hashlib.sha256(np.ascontiguousarray(d['row'][fit]).tobytes()).hexdigest(),
 source_sha256=sha(__file__),model_sha256=models,predictions_sha256=sha(run/'predicciones_held_Train.npz'),
 Validation_access=False,Test_access=False,actual_data_files_accessed=sorted(access),
 conclusion='Gate passed: justify one matched full primary trial.' if passed else 'Gate failed: do not spend a primary neural trial on this cycle bank.',
 seconds=time.time()-start))
print(json.dumps(dict(stage='complete',passed=passed,comparison=comparison,seconds=time.time()-start),default=lambda q:q.tolist() if isinstance(q,np.ndarray) else q),flush=True)
