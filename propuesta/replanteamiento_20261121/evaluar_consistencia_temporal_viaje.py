"""Full unchanged primary OD evaluation of aggregate-trip calibration; frozen settings."""
from base_replanteo import *
import gc,shutil
from inferencia import feature_input
from catboost import CatBoostRegressor
access=deny_test();start=time.time();run=OUT/'consistencia_temporal_viaje_s0';folder=run/'etapa_B';folder.mkdir(exist_ok=True)
stage=read(run/'resultado_modelo.json');sm=read(run/'resumenes_publicos/resultado.json');assert stage['complete'] and sm['complete']
previous=read(OUT/'progreso_vial_recurrente_s0/etapa_B/resultado.json');assert previous['complete']
sources=['evaluar_consistencia_temporal_viaje.py','base_replanteo.py']
if not (folder/'fuentes').exists():
 (folder/'fuentes').mkdir()
 for n in sources:shutil.copy2(OUT/n,folder/'fuentes'/n)
else:
 for n in sources:assert sha(OUT/n)==sha(folder/'fuentes'/n)
plan=dict(iterations=1000,depth=8,learning_rate=.05,l2_leaf_reg=5,seed=20261122,border_count=128,
 GPU_fraction=.35,CPU_threads=2,input=246,target='OriginalT/600duringfit; forecasts*600 evaluatedonoriginalTseconds.',
 primary_Train=1023485,Validation=127941,Test='sealed',main_model='full246_MAE',
 selection='Fixedfinalpointmodel; no choosingepochs/alphas. No new216controlfit: reuseexactsamepublishedpreviouscontrol.',
 uncertainty_trigger='Ifmainpointmodelnewnumericalbest, fitq10/q90foritsuncertainty; MAEpointservesasthemedian. Otherwisepreservepointcomparisononly.',
 gate='>=.5secondsbetterthanprotectedbestandpreviousuncalibrated246model; upperdayCI<0 forboth.')
if (folder/'protocolo.json').exists():assert read(folder/'protocolo.json')['plan']==plan
else:write(folder/'protocolo.json',dict(plan=plan,reference=read(OUT/'nuevo_mejor_verificado.json')))
reference=read(folder/'protocolo.json')['reference']
tr,X,names=feature_input('train',read(BEST/'resultado.json'));va,V,vnames=feature_input('validation',read(BEST/'resultado.json'));assert names==vnames
summ={}
for split,d in [('train',tr),('validation',va)]:
 p=run/'resumenes_publicos'/(split+'_resumenes.npz');assert sha(p)==sm['outputs'][split]['sha256']
 a=dict(np.load(p))
 for k in ['row','timestamp']:assert np.array_equal(a[k],d[k])
 summ[split]=a['summary']
X=np.column_stack([X,summ['train']]);V=np.column_stack([V,summ['validation']]);del summ,a;gc.collect()
y=np.load(DATA/'train_y.npy');truth=np.load(DATA/'validation_y.npy')
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index']
assert len(eligible)==1023485 and len(truth)==127941 and np.max(tr['timestamp'][eligible]+y[eligible])<va['timestamp'].min()
prev=dict(np.load(OUT/'progreso_vial_recurrente_s0/etapa_B/validacion.npz'))
for k in ['row','timestamp']:assert np.array_equal(prev[k],va[k])
assert np.array_equal(prev['true'],truth)
ref=dict(np.load(OUT/reference['id']/'validacion.npz'))
assert np.array_equal(ref['true'],truth)
for k in ['row','timestamp']:assert np.array_equal(ref[k],va[k])
common=dict(iterations=1000,depth=8,learning_rate=.05,l2_leaf_reg=5,random_seed=20261122,
 thread_count=2,task_type='GPU',devices='0',gpu_ram_part=.35,border_count=128,allow_writing_files=False,verbose=False)
manifestfile=folder/'modelos_completados.json';manifest=read(manifestfile) if manifestfile.exists() else {}
preds={};metrics={}
def fit_predict(name,loss):
 p=folder/(name+'.cbm')
 if name in manifest:
  assert sha(p)==manifest[name]['sha256'];m=CatBoostRegressor();m.load_model(str(p))
 else:
  assert not p.exists(),'Uncommittedcheckpointrequiresinspection'
  m=CatBoostRegressor(**common,loss_function=loss);m.fit(X[eligible],y[eligible].astype(np.float64)/600);m.save_model(str(p))
  manifest[name]=dict(file=p.name,sha256=sha(p),bytes=p.stat().st_size,loss=loss,width=246);write(manifestfile,manifest)
 pred=(m.predict(V,thread_count=2)*600).clip(60,1785)
 assert np.array_equal((m.predict(V[:17][::-1],thread_count=2)*600).clip(60,1785)[::-1],pred[:17])
 assert np.array_equal((m.predict(V[:1],thread_count=2)*600).clip(60,1785),pred[:1])
 preds[name]=pred;metrics[name]=metric(truth,pred)
 np.savez_compressed(folder/(name+'_predicciones.npz'),row=va['row'],timestamp=va['timestamp'],true=truth,pred=pred)
 write(folder/(name+'_resultado.json'),dict(validation=metrics[name],model=manifest[name],
  predictions_sha256=sha(folder/(name+'_predicciones.npz')),Test=False))
 print(json.dumps(dict(stage='complete_'+name,validation=metrics[name],seconds=time.time()-start)),flush=True)
fit_predict('full246_MAE','MAE')
is_new=metrics['full246_MAE']['mae_s']<reference['validation']['mae_s']
if is_new:
 fit_predict('full246_q10','Quantile:alpha=0.1');fit_predict('full246_q90','Quantile:alpha=0.9')
 quantiles=np.sort(np.column_stack([preds['full246_q10'],preds['full246_MAE'],preds['full246_q90']]),axis=1)
else:quantiles=np.empty((len(truth),0),np.float64)
comparisons=dict(full_vs_reference=paired(truth,preds['full246_MAE'],ref['pred'],va['timestamp']),
 full_vs_uncalibrated=paired(truth,preds['full246_MAE'],prev['full246_MAE'],va['timestamp']),
 full_vs_216control=paired(truth,preds['full246_MAE'],prev['control216_MAE'],va['timestamp']))
gate=all(comparisons[k]['delta_mae_s']<=-.5 and comparisons[k]['ci95_s'][1]<0 for k in ['full_vs_reference','full_vs_uncalibrated'])
np.savez_compressed(folder/'validacion.npz',row=va['row'],timestamp=va['timestamp'],true=truth,pred=preds['full246_MAE'],
 quantiles=quantiles,reference=ref['pred'],uncalibrated=prev['full246_MAE'],control216=prev['control216_MAE'],**preds)
models={}
main_qualified=metrics['full246_MAE']['mae_s']<131.06 or is_new
for name,info in manifest.items():
 keep=main_qualified or metrics[name]['mae_s']<131.06 or metrics[name]['mae_s']<reference['validation']['mae_s']
 models[name]=dict(**info,validation=metrics[name],preserved=bool(keep),requires_GRU=bool(keep))
result=dict(complete=True,primary_validation=True,metrics=metrics,models=models,reference_id=reference['id'],
 reference=reference['validation'],comparisons=comparisons,main_validation=metrics['full246_MAE'],main_model='full246_MAE',
 meaningful_gate_passed=bool(gate),new_numerical_minimum=bool(is_new),sequence_model_sha256=stage['model_sha256'],
 reused_216_control=previous['metrics']['control216_MAE'],no_duplicate_control_training=True,
 no_query_private_fields=True,quantile_models_fit=bool(is_new),
 predictions_sha256=sha(folder/'validacion.npz'),sources={n:sha(folder/'fuentes'/n) for n in sources},
 all_completed_metrics_predictions_retained=True,Test_access=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(folder/'resultado.json',result)
removed=[]
for name,info in models.items():
 p=folder/info['file']
 if not info['preserved']:
  assert p.resolve().is_relative_to(folder.resolve()) and sha(p)==info['sha256'];p.unlink();removed.append(info)
 dup=folder/(name+'_predicciones.npz');z=np.load(dup);assert np.array_equal(z['pred'],preds[name]);z.close()
 assert dup.resolve().is_relative_to(folder.resolve());dup.unlink()
keep=any(v['preserved'] for v in models.values())
if not keep:
 p=Path(stage['model_file']);assert p.resolve().is_relative_to(run.resolve()) and sha(p)==stage['model_sha256'];p.unlink()
 init=OUT/'inicializacion_consistencia_temporal_s0';p=init/'inicial.pt'
 im=read(init/'resultado.json');assert p.resolve().is_relative_to(init.resolve()) and sha(p)==im['model_sha256'];p.unlink()
write(folder/'limpieza.json',dict(unqualified_weights_removed=removed,sequence_and_initialization_preserved=bool(keep),
 all_completed_predictions_metrics_sources_retained=True))
print(json.dumps(dict(complete=True,main=metrics['full246_MAE'],comparisons=comparisons,gate=gate,new_numerical_minimum=is_new),default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else str(v)),flush=True)
