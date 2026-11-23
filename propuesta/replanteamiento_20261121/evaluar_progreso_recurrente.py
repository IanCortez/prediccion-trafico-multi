"""Fixed matched full-primary tree comparison on frozen public-only GRU summaries."""
from base_replanteo import *
import gc,shutil
from catboost import CatBoostRegressor
from inferencia import feature_input
access=deny_test();start=time.time();run=OUT/'progreso_vial_recurrente_s0';folder=run/'etapa_B';folder.mkdir(exist_ok=True)
stage=read(run/'resultado_etapa_A.json');summaryfolder=run/'resumenes_publicos';sm=read(summaryfolder/'resultado.json')
assert stage['complete'] and sm['complete'] and sm['model_sha256']==stage['model_sha256']
sources=['evaluar_progreso_recurrente.py','base_replanteo.py']
if not (folder/'fuentes').exists():
 (folder/'fuentes').mkdir()
 for n in sources:shutil.copy2(OUT/n,folder/'fuentes'/n)
else:
 for n in sources:assert sha(OUT/n)==sha(folder/'fuentes'/n)
reference=read(run/'protocolo.json')['reference']
plan=dict(seed=20261122,iterations=1000,depth=8,learning_rate=.05,l2_leaf_reg=5,
 GPU_fraction=.35,border_count=128,CPU_threads=2,selection='FixedMAEpointmodel with246features; no selectionbetweenalphas/control/epochs.',
 models=['control216_MAE','full246_MAE','full246_q10','full246_q50','full246_q90'],
 input='Original216legalraw5derivedfeatures, plus30frozenpredicted-route/time summaries. No actualqueryroad,N,elapsed,T.',
 target='OriginalTseconds dividedby600onlyduringfit; forecasts*600 and clippedtooriginal60..1785support.',
 primary_Train=1023485,Validation=127941,Test='sealed',
 gate='Full246_MAE at least.5seconds overprotectedbest andmatched216control, both upperdayCI<0. Then certifyraw5/replay andmatchedprefix-resetcontrol.',
 preservation='AllfivefullValidationmetrics/predictions including above131.06; allqualifiedweights<131.06 orstrictglobalbest. Full246models depend on frozenGRU.')
if (folder/'protocolo.json').exists():assert read(folder/'protocolo.json')==plan
else:write(folder/'protocolo.json',plan)
tr,X,names=feature_input('train',read(BEST/'resultado.json'));va,V,vnames=feature_input('validation',read(BEST/'resultado.json'))
assert names==vnames and len(names)==216
banks={}
for split,data in [('train',tr),('validation',va)]:
 p=summaryfolder/(split+'_resumenes.npz');assert sha(p)==sm['outputs'][split]['sha256']
 a=dict(np.load(p));assert np.array_equal(a['row'],data['row']) and np.array_equal(a['timestamp'],data['timestamp'])
 assert a['summary'].shape==(len(data['row']),30) and np.isfinite(a['summary']).all()
 banks[split]=a['summary']
X=np.column_stack([X,banks['train']]);V=np.column_stack([V,banks['validation']]);del banks;gc.collect()
y=np.load(DATA/'train_y.npy');truth=np.load(DATA/'validation_y.npy')
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index']
assert len(eligible)==1023485 and len(truth)==127941 and np.max(tr['timestamp'][eligible]+y[eligible])<va['timestamp'].min()
common=dict(iterations=1000,depth=8,learning_rate=.05,l2_leaf_reg=5,random_seed=20261122,
 thread_count=2,task_type='GPU',devices='0',gpu_ram_part=.35,border_count=128,allow_writing_files=False,verbose=False)
manifestfile=folder/'modelos_completados.json';manifest=read(manifestfile) if manifestfile.exists() else {}
metrics={};preds={};saved={}
for name,loss,width in [('control216_MAE','MAE',216),('full246_MAE','MAE',246),
 ('full246_q10','Quantile:alpha=0.1',246),('full246_q50','Quantile:alpha=0.5',246),('full246_q90','Quantile:alpha=0.9',246)]:
 file=folder/(name+'.cbm')
 if name in manifest:
  assert sha(file)==manifest[name]['sha256'];model=CatBoostRegressor();model.load_model(str(file))
 else:
  assert not file.exists(),'Uncommittedweightrequiresinspection'
  model=CatBoostRegressor(**common,loss_function=loss)
  model.fit(X[eligible,:width],y[eligible].astype(np.float64)/600)
  model.save_model(str(file));manifest[name]=dict(file=file.name,sha256=sha(file),bytes=file.stat().st_size,loss=loss,width=width)
  write(manifestfile,manifest)
 pred=(model.predict(V[:,:width],thread_count=2)*600).clip(60,1785)
 reverse=(model.predict(V[:17,:width][::-1],thread_count=2)*600).clip(60,1785)[::-1]
 single=(model.predict(V[:1,:width],thread_count=2)*600).clip(60,1785)
 assert np.array_equal(pred[:17],reverse) and np.array_equal(pred[:1],single)
 preds[name]=pred;metrics[name]=metric(truth,pred)
 # Preserve each completed output immediately, independent of later jobs succeeding.
 np.savez_compressed(folder/(name+'_predicciones.npz'),row=va['row'],timestamp=va['timestamp'],true=truth,pred=pred)
 write(folder/(name+'_resultado.json'),dict(validation=metrics[name],model=manifest[name],
  predictions_sha256=sha(folder/(name+'_predicciones.npz')),actual_labels_evaluated_in_seconds=True,Test=False))
 print(json.dumps(dict(stage='complete_'+name,validation=metrics[name],seconds=time.time()-start)),flush=True)
 del model;gc.collect()
q=np.column_stack([preds['full246_q10'],preds['full246_q50'],preds['full246_q90']])
quantiles=np.sort(q,axis=1)
ref=dict(np.load(OUT/reference['id']/'validacion.npz'))
for key in ['row','timestamp']:assert np.array_equal(ref[key],va[key])
assert np.array_equal(ref['true'],truth)
comp=dict(full_vs_reference=paired(truth,preds['full246_MAE'],ref['pred'],va['timestamp']),
 full_vs_control=paired(truth,preds['full246_MAE'],preds['control216_MAE'],va['timestamp']))
gate=all(z['delta_mae_s']<=-.5 and z['ci95_s'][1]<0 for z in comp.values())
np.savez_compressed(folder/'validacion.npz',row=va['row'],timestamp=va['timestamp'],true=truth,
 pred=preds['full246_MAE'],quantiles=quantiles,raw_quantiles=q,reference=ref['pred'],**preds)
main_qualified=metrics['full246_MAE']['mae_s']<131.06 or metrics['full246_MAE']['mae_s']<reference['validation']['mae_s']
for name,info in manifest.items():
 keep=metrics[name]['mae_s']<131.06 or metrics[name]['mae_s']<reference['validation']['mae_s'] or (main_qualified and name.startswith('full246'))
 saved[name]=dict(**info,validation=metrics[name],preserved=bool(keep),requires_GRU=bool(keep and name.startswith('full246')))
result=dict(complete=True,primary_validation=True,metrics=metrics,reference=reference['validation'],comparisons=comp,
 meaningful_gate_passed=bool(gate),main_model='full246_MAE',main_validation=metrics['full246_MAE'],
 models=saved,sequence_model_sha256=stage['model_sha256'],public_summary_manifest_sha256=sha(summaryfolder/'resultado.json'),
 predictions_sha256=sha(folder/'validacion.npz'),sources={n:sha(folder/'fuentes'/n) for n in sources},
 all_completed_metrics_predictions_retained=True,Test_access=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(folder/'resultado.json',result)
# Only exact duplicates / unqualified weights, after all predictions and hashes verified.
removed=[]
for name,entry in saved.items():
 p=folder/entry['file']
 if not entry['preserved']:
  assert p.resolve().is_relative_to(folder.resolve()) and sha(p)==entry['sha256'];p.unlink();removed.append(entry)
 duplicate=folder/(name+'_predicciones.npz')
 z=np.load(duplicate);assert np.array_equal(z['pred'],preds[name]);z.close()
 assert duplicate.resolve().is_relative_to(folder.resolve());duplicate.unlink()
keep_GRU=any(z['requires_GRU'] for z in saved.values()) or stage['root_diagnostic_validation']['mae_s']<131.06
if not keep_GRU:
 p=Path(stage['model_file']);assert p.resolve().is_relative_to(run.resolve()) and sha(p)==stage['model_sha256'];p.unlink()
write(folder/'limpieza.json',dict(removed_weights=removed,sequence_weights_preserved=bool(keep_GRU),
 all_completed_predictions_metrics_sources_preserved=True,duplicate_point_archives_removed_after_exactproof=True))
print(json.dumps(dict(stage='complete_full_primary',metrics=metrics,comparisons=comp,gate=gate,seconds=time.time()-start)),flush=True)
