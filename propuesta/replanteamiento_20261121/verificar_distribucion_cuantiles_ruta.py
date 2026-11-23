"""Analytical mixturequantile cases and exact chronological dataset integrity."""
from base_replanteo import *
from numba import set_num_threads
from distribucion_cuantiles_ruta import mixture_quantiles,rearrange
from catboost import CatBoostRegressor
access=deny_test();start=time.time();set_num_threads(2)
alpha=np.arange(1,10)/10;uniform=60+1725*alpha;values=np.broadcast_to(uniform,(3,2,9)).copy();weights=np.array([[1.,0.],[.3,.7],[0.,1.]])
q=mixture_quantiles(values,weights);expected=60+1725*np.array([.1,.5,.9]);assert np.max(np.abs(q-expected))<1e-6
point=np.empty((2,2,9));point[:,0]=300;point[:,1]=900;w=np.array([[.75,.25],[.25,.75]])
p=mixture_quantiles(point,w);assert abs(p[0,1]-300)<1e-6 and abs(p[1,1]-900)<1e-6 and np.all(p[:,1:]>=p[:,:-1])
assert np.array_equal(p,mixture_quantiles(point[::-1],w[::-1])[::-1])
source=OUT/'oraculo_tiempo_calle_train_s0';r=read(source/'resultado.json');assert sha(source/'auxiliares_Train.npz')==r['auxiliary_sha256']
with np.load(source/'auxiliares_Train.npz') as z:d={key:z[key] for key in ['row','train_index','timestamp','true']}
x=dict(np.load(DATA/'train_x.npz'));y=np.load(DATA/'train_y.npy');ix=d['train_index']
assert np.array_equal(x['row'][ix],d['row']) and np.array_equal(x['timestamp'][ix],d['timestamp']) and np.array_equal(y[ix],d['true'])
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index'];cut=int(x['timestamp'][eligible].max())-21*86400
assert np.all(d['timestamp'][:40000]+d['true'][:40000]<cut) and np.all(d['timestamp'][40000:]>=cut)
# Inspect actual installedlossdefaults without touchingheldlabels or activefit.
probe=CatBoostRegressor(iterations=2,depth=2,loss_function='MultiQuantile:alpha=0.1,0.5,0.9',allow_writing_files=False,verbose=False,thread_count=1)
probe.fit(np.arange(64).reshape(-1,1),np.linspace(60,1785,64));params=probe.get_all_params()
result=dict(passed=True,uniform_quantiles_exact_to_1e_6=True,mixture_mass_median_and_order_exact_to_1e_6=True,
 dataset_rows_timestamps_original_labels_exact=True,fit_arrival_purge_verified=True,held_cutoff=cut,
 CatBoost_MultiQuantile_leaf_method=params.get('leaf_estimation_method'),no_Validation_or_Test_access=True,
 source_sha256=sha(OUT/'distribucion_cuantiles_ruta.py'),seconds=time.time()-start)
write(OUT/'verificacion_distribucion_cuantiles_ruta.json',result);print(json.dumps(result),flush=True)
