"""Time-specific chronological Train gate for ordered categorical OD statistics."""
from base_replanteo import *
import pandas as pd,shutil
from catboost import CatBoostRegressor
from inferencia import feature_input
access=deny_test();start=time.time();run=OUT/'piloto_categorias_od_train_s0';assert not run.exists();run.mkdir();shutil.copy2(__file__,run/'fuente_ejecutada.py')
baseline=OUT/'piloto_reloj_fino_train_s0';prior=read(baseline/'resultado.json');base_pred=dict(np.load(baseline/'predicciones_held_Train.npz'))
plan=dict(scope='Chronological heldTrain prediction gate, not primaryValidation. Distinct from numeric-only CatBoost and learned geometric memory.',
 hypothesis='Explicit multiscale categorical OD interactions with ordered target statistics may capture frequent-trip conditional duration distributions that numeric coordinate splits miss.',
 seed=20261122,fit_n=200000,held_n=50000,held_last_days=21,public_numeric_features=206,
 categorical_scales_m=[64,128,256],pair_scales_m=[128,256],time_pair='256mODpair crossed with weekday and6hourdeparture block',
 control='Reuse exact stored numeric206control from fine-clock gate: identicalfit/held rows,params and features. Verify rows/labels/timestamps/fit-row hash.',
 max_ctr_complexity=1,ctr_leaf_count_limit=200000,iterations=800,depth=7,learning_rate=.05,l2_leaf_reg=5,
 checkpoint='Fixed final800trees, no held stopping or parameter search.',
 gate='At least1second heldTrain MAE improvement and upper95percent daybootstrap deltaCI<0; otherwise no full primary trial.',
 query=['o_lon','o_lat','d_lon','d_lat','timestamp'],Validation=False,Test=False,CPU_threads=2,model_file='Save only if gate passes; always keep metrics/predictions/source.')
write(run/'plan.json',plan)
rec=read(BEST/'resultado.json');d,X,names=feature_input('train',rec);keep=np.array([not n.startswith('mm_') for n in names]);assert keep.sum()==206
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index'];y=np.load(DATA/'train_y.npy');ts=d['timestamp'].astype(np.int64)
split=int(ts[eligible].max())-21*86400;fit_pool=eligible[ts[eligible]+y[eligible]<split];held_pool=eligible[ts[eligible]>=split]
rng=np.random.default_rng(20261122);fit=np.sort(rng.choice(fit_pool,200000,replace=False));held=np.sort(rng.choice(held_pool,50000,replace=False));ix=np.concatenate([fit,held])
assert hashlib.sha256(np.ascontiguousarray(d['row'][fit]).tobytes()).hexdigest()==prior['selected_fit_rows_sha256']
for key in ['row','timestamp']:assert np.array_equal(base_pred[key],d[key][held])
assert np.array_equal(base_pred['true'],y[held])
frame=pd.DataFrame(X[ix][:,keep],columns=[n for n,k in zip(names,keep) if k]);del X
sx=111195*np.cos(np.deg2rad(41.16));coordinates={}
for endpoint in ['o','d']:
 coordinates[endpoint]=np.column_stack([(d[endpoint+'_lon'][ix]+8.62)*sx,(d[endpoint+'_lat'][ix]-41.16)*111195])
categories=[]
for size in [64,128,256]:
 values={}
 for endpoint in ['o','d']:
  cell=np.floor(coordinates[endpoint]/size).astype(np.int32)
  name='category_'+endpoint+'_'+str(size);values[endpoint]=pd.Series(cell[:,0].astype(str))+'_'+pd.Series(cell[:,1].astype(str))
  frame[name]=values[endpoint];categories.append(name)
 if size in [128,256]:
  name='category_OD_'+str(size);frame[name]=values['o']+'>'+values['d'];categories.append(name)
local=pd.to_datetime(ts[ix],unit='s',utc=True).tz_convert('Europe/Lisbon')
name='category_OD256_weekday_daypart';frame[name]=frame['category_OD_256']+'|'+pd.Series(np.asarray(local.dayofweek).astype(str))+'|'+pd.Series((np.asarray(local.hour)//6).astype(str));categories.append(name)
params=dict(prior['plan']['model']);params.update(max_ctr_complexity=1,ctr_leaf_count_limit=200000)
assert params['iterations']==800 and params['depth']==7 and params['thread_count']==2 and not params['allow_writing_files']
model=CatBoostRegressor(**params);model.fit(frame.iloc[:200000],y[fit],cat_features=categories,verbose=False)
pred=model.predict(frame.iloc[200000:]);m=metric(y[held],pred);control=metric(y[held],base_pred['control'])
assert control==prior['metrics']['control']
comparison=paired(y[held],pred,base_pred['control'],ts[held]);comparison['interpretation']='Chronological heldTrain only; no Validation labels or adaptive stopping. Control reused exactly from stored predictions.'
passed=bool(comparison['delta_mae_s']<=-1 and comparison['ci95_s'][1]<0)
if passed:model.save_model(str(run/'categorical.cbm'))
np.savez_compressed(run/'predicciones_held_Train.npz',row=d['row'][held],timestamp=ts[held],true=y[held],categorical=pred,control=base_pred['control'])
result=dict(passed=passed,metrics=m,control=control,comparison=comparison,categorical_features=categories,
 model_sha256=sha(run/'categorical.cbm') if passed else None,source_sha256=sha(run/'fuente_ejecutada.py'),
 baseline_predictions_sha256=sha(baseline/'predicciones_held_Train.npz'),predictions_sha256=sha(run/'predicciones_held_Train.npz'),
 all_fit_arrivals_before_held=True,Validation_access=False,Test_access=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(run/'resultado.json',result)
print(json.dumps(dict(passed=passed,heldTrain_validation=m,control=control,comparison=comparison,seconds=result['seconds']),default=lambda q:q.tolist() if isinstance(q,np.ndarray) else q),flush=True)
