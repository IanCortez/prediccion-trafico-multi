"""LocalGPU quantile/categorical capability and resource preflight, no realdatasets."""
from base_replanteo import *
from catboost import CatBoostRegressor
import pandas as pd
access=deny_test();start=time.time();rng=np.random.default_rng(20261122);X=rng.normal(size=(512,4)).astype(np.float32)
frame=pd.DataFrame(X,columns=['x0','x1','x2','x3']);frame['code']=(np.arange(512)%17).astype(str)
target=np.clip(.8+.2*X[:,0]+.03*(np.arange(512)%17),.1,2.975)
params=dict(iterations=32,depth=6,learning_rate=.05,loss_function='Quantile:alpha=0.5',random_seed=20261122,thread_count=2,
 task_type='GPU',devices='0',gpu_ram_part=.25,one_hot_max_size=32,allow_writing_files=False,verbose=False)
model=CatBoostRegressor(**params);model.fit(frame,target,cat_features=['code']);pred=model.predict(frame.iloc[:17])
assert pred.shape==(17,) and np.isfinite(pred).all()
r=dict(passed=True,GPU_quantile_with_categorical17_supported=True,gpu_ram_part=.25,task_type=model.get_all_params()['task_type'],
 source_sha256=sha(__file__),datasets_opened=False,Test_access=False,seconds=time.time()-start)
write(OUT/'verificacion_catboost_cuantiles_gpu.json',r);print(json.dumps(r),flush=True)
