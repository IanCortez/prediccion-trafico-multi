"""Repair only a Train-fitted scalar calibration; all input banks remain immutable."""
from base_replanteo import *
access=deny_test();start=time.time();run=OUT/'datos_prototipos_historicos_s0'
old=dict(np.load(run/'constructor_Train.npz'));H=np.load(run/'train_historicos.npz')['history']
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index'];y=np.load(DATA/'train_y.npy')
mass=H[eligible,:,18].astype(np.float64);prior=H[eligible,:,16].astype(np.float64)
best=mass.argmax(1);x=prior[np.arange(len(eligible)),best]*600
valid=(mass.max(1)>0)&(x>0);xx=np.log(x[valid].astype(np.float64));yy=np.log(y[eligible[valid]].astype(np.float64))
assert np.std(xx)>.05
coeff,residual,rank,singular,rcond=np.polyfit(xx,yy,1,full=True);beta,intercept=coeff
intercept+=np.log(np.median(y[eligible[valid]].astype(float)/np.exp(intercept+beta*xx)))
assert rank==2 and beta>0 and np.isfinite(coeff).all()
path=run/'constructor_Train_fp64.npz';assert not path.exists()
np.savez_compressed(path,mean=old['mean'],std=old['std'],time_anchor=np.array([intercept,beta],np.float32),n=valid.sum(),normalization_routes=old['normalization_routes'])
record=dict(reason='NumPy polyfit float32 default rcond scales with about1million rows and truncated the fit. Refit logs in float64 before any model training.',
 old_constructor_sha256=sha(run/'constructor_Train.npz'),old_time_anchor=old['time_anchor'],
 used_constructor_file=path.name,used_constructor_sha256=sha(path),new_time_anchor=[intercept,beta],
 log_time_std=float(np.std(xx)),rank=int(rank),rcond=float(rcond),singular_values=singular,
 normalization_mean_std_unchanged=True,all_route_input_cache_files_unchanged=True,
 scope='EligibleTrain only, supervised scalar model preprocessing; no Validation labels.',
 source_sha256=sha(__file__),Test_access=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(run/'calibracion_fp64.json',record)
print(json.dumps(dict(corrected=True,rank=int(rank),old_anchor=old['time_anchor'].tolist(),new_anchor=[float(intercept),float(beta)],Train_n=int(valid.sum()))))
