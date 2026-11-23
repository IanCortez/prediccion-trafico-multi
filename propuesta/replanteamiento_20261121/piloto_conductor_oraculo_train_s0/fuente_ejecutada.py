"""Privileged-driver chronological Train-only oracle; never a permitted OD input."""
from base_replanteo import *
import csv,shutil,pandas as pd
from catboost import CatBoostRegressor
from inferencia import feature_input
access=deny_test();start=time.time();run=OUT/'piloto_conductor_oraculo_train_s0';assert not run.exists();run.mkdir();shutil.copy2(__file__,run/'fuente_ejecutada.py')
baseline=OUT/'piloto_reloj_fino_train_s0';prior=read(baseline/'resultado.json');control=dict(np.load(baseline/'predicciones_held_Train.npz'))
plan=dict(scope='Privilegeddriver chronological heldTrainoracle, not deployableOD. Train metadata supervision idea only.',
 hypothesis='Driver behavior might explain residualtime beyond OD. A true-driver oracle must showmeaningfulsignal before any latentdriver architecture.',
 fit_n=200000,held_n=50000,held_last_days=21,seed=20261122,
 input='206publicnumericODcontrol plusone categorical actualTAXI_ID, used onlyin thisprivilegedoracle. Neveraddhelddriver to primaryValidation/Test.',
 metadata='ReadbinaryCSVmembership inselectedTrain rows beforedecode; no Validation/Test metadata decoded. Persist only stablehasheddriverlabels.',
 model='Exact800tree depth7MAEpriorcontrol parameters; max_ctr_complexity1,ctr_leaf_count_limit200000, no heldstopping.',
 gate='At least3seconds heldTrainMAE improvement andupper95percentdaybootstrapdeltaCI<0; otherwise no latentdriver training.',
 caution='Even positiveoracle would not showdriver inferablefromfivequeryfields; requiresseparateposteriorpredictability test.',
 resources='CPU2; noGPU, <=100MiB artifacts.',Validation=False,Test=False)
write(run/'plan.json',plan)
d,X,names=feature_input('train',read(BEST/'resultado.json'));keep=np.asarray([not n.startswith('mm_') for n in names]);assert keep.sum()==206
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index'];y=np.load(DATA/'train_y.npy');ts=d['timestamp'].astype(np.int64)
split=int(ts[eligible].max())-21*86400;fp=eligible[ts[eligible]+y[eligible]<split];hp=eligible[ts[eligible]>=split];rng=np.random.default_rng(20261122)
fit=np.sort(rng.choice(fp,200000,replace=False));held=np.sort(rng.choice(hp,50000,replace=False));indices=np.r_[fit,held];rows=d['row'][indices]
assert hashlib.sha256(np.ascontiguousarray(d['row'][fit]).tobytes()).hexdigest()==prior['selected_fit_rows_sha256']
for key in ['row','timestamp']:assert np.array_equal(control[key],d[key][held])
assert np.array_equal(control['true'],y[held])
lookup=np.full(int(rows.max())+1,-1,np.int32);lookup[rows]=np.arange(len(rows),dtype=np.int32);labels=np.full(len(rows),'',dtype='<U16');found=0
with (ROOT/'tesis/porto/porto/train.csv').open('rb') as f:
 header=next(csv.reader([next(f).decode('utf-8-sig').strip()]));driver_col=header.index('TAXI_ID');time_col=header.index('TIMESTAMP')
 for rawrow,line in enumerate(f):
  if rawrow>=len(lookup):break
  j=int(lookup[rawrow])
  if j<0:continue
  fields=next(csv.reader([line.decode('utf-8')]));assert int(fields[time_col])==ts[indices[j]]
  value=fields[driver_col];assert value.strip();labels[j]=hashlib.sha256(value.encode('utf-8')).hexdigest()[:16];found+=1
assert found==len(rows) and np.all(labels!='')
frame=pd.DataFrame(X[indices][:,keep],columns=[n for n,k in zip(names,keep) if k]);del X
frame['privileged_driver']=labels
params=dict(prior['plan']['model']);params.update(max_ctr_complexity=1,ctr_leaf_count_limit=200000)
model=CatBoostRegressor(**params);model.fit(frame.iloc[:200000],y[fit],cat_features=['privileged_driver'],verbose=False);pred=model.predict(frame.iloc[200000:])
comparison=paired(y[held],pred,control['control'],ts[held]);comparison['interpretation']='PrivilegeddriverchronologicalheldTrain only; no primaryODdriver input or confirmation.'
passed=bool(comparison['delta_mae_s']<=-3 and comparison['ci95_s'][1]<0)
if passed:model.save_model(str(run/'oraculo_conductor.cbm'))
np.savez_compressed(run/'etiquetas_conductor_Train.npz',row=rows,timestamp=ts[indices],driver_hash=labels,fit_n=200000)
np.savez_compressed(run/'predicciones_held_Train.npz',row=d['row'][held],timestamp=ts[held],true=y[held],pred=pred,control=control['control'])
result=dict(passed=passed,scope=plan['scope'],metrics=metric(y[held],pred),control=metric(y[held],control['control']),comparison=comparison,
 unique_fit_drivers=int(np.unique(labels[:200000]).size),unique_held_drivers=int(np.unique(labels[200000:]).size),
 source_sha256=sha(run/'fuente_ejecutada.py'),predictions_sha256=sha(run/'predicciones_held_Train.npz'),hashed_labels_sha256=sha(run/'etiquetas_conductor_Train.npz'),
 Validation_metadata_decoded=False,Test_metadata_decoded=False,seconds=time.time()-start)
write(run/'resultado.json',result);print(json.dumps(dict(passed=passed,metrics=result['metrics'],control=result['control'],delta_mae_s=comparison['delta_mae_s'],seconds=result['seconds'])),flush=True)
