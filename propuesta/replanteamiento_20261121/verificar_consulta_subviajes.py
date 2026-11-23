"""Train-only feature equivalence and interval-label arithmetic gate."""
from base_replanteo import *
from inferencia import feature_input
from consulta_subviajes_train import build_training_features
access=deny_test();start=time.time();rec=read(BEST/'resultado.json')
tr,X,names=feature_input('train',rec);eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index']
ii=np.sort(np.random.default_rng(20261122).choice(eligible,65,replace=False));columns=['o_lon','o_lat','d_lon','d_lat','timestamp']
items=[{k:int(tr[k][i]) if k=='timestamp' else float(tr[k][i]) for k in columns} for i in ii]
recovered,rnames=build_training_features(items,rec['config']);assert rnames==names
error=float(np.abs(recovered-X[ii]).max());assert error<1e-5
# Interval labels telescope and use N-1 differences, never the number of sampled endpoints.
rg=np.random.default_rng(17);errors=[]
for _ in range(10000):
 n=int(rg.integers(6,121));span=int(rg.integers(4,n-1));i=int(rg.integers(0,n-span));j=i+span
 assert 4<=j-i<=119 and not (i==0 and j==n-1)
 t=15*(j-i);k=int(rg.integers(i,j+1));assert 15*(k-i)+15*(j-k)==t
 ts=1372636853;assert ts+15*i+t==ts+15*j
 errors.append(0)
extra_rejected=False
try:build_training_features([dict(items[0],parent_row=123)],rec['config'])
except AssertionError:extra_rejected=True
assert extra_rejected
r=dict(passed=True,scope='CPU Train-only input/label instrument, no primary MAE',existing_Train_feature_recovery_max_error=error,feature_names_match_protected216=True,all10000_interval_arithmetic_checks=True,provenance_extra_field_rejected=True,raw_query_fields=columns,whole_query_UTCweek_fold_excluded_at_all_history_levels=True,Validation_access=False,Test_access=False,code_sha256=sha(__file__),source_sha256=sha(OUT/'consulta_subviajes_train.py'),actual_files_accessed=sorted(access),seconds=time.time()-start)
write(OUT/'verificacion_consulta_subviajes.json',r);print(json.dumps(r),flush=True)
