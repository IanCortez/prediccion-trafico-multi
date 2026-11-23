"""Audit negative direct-MAE trial and remove only exact discarded intermediates."""
from base_replanteo import *
r=read(OUT/'mediana_directa_s0/resultado.json');run=OUT/r['id'];z=dict(np.load(run/'validacion.npz'))
ref=read(OUT/'historica_fresca_s0/resultado.json');base=dict(np.load(OUT/ref['id']/'validacion.npz'))
assert r['epochs_run']==6 and not r['learned_update_selected'] and r['best_epoch']==0 and r['model_file'] is None
for key in ['row','timestamp','true']:assert np.array_equal(z[key],base[key])
assert np.array_equal(z['pred'],base['pred']) and metric(z['true'],z['pred'])==r['validation']
for row in r['all_epoch_validation']:
 e=row['epoch'];assert metric(z['true'],z['epoch_'+str(e)])==row['validation']
 assert metric(z['true'],z['continuous_epoch_'+str(e)])==row['continuous_validation']
best=min(r['all_epoch_validation'],key=lambda q:q['validation']['mae_s']);last=r['all_epoch_validation'][-1]
for name,digest in r['sources'].items():assert sha(run/'fuentes'/name)==digest
archive=run/'recuperacion_memoria_20261122';old=dict(np.load(archive/'predicciones_avance.npz'))
for key,value in old.items():assert np.array_equal(value,z[key]),key
assert read(archive/'protocolo.json')==r['config']
result=dict(learned_improvement=False,selected_epoch=0,selected_output_is_existing_verified_reference=True,
 best_learned_epoch=best['epoch'],best_learned_validation=best['validation'],final_epoch_validation=last['validation'],
 best_learned_minus_reference=paired(z['true'],z['epoch_'+str(best['epoch'])],base['pred'],z['timestamp']),
 final_minus_reference=paired(z['true'],z['epoch_6'],base['pred'],z['timestamp']),
 Train_loss_first=r['all_epoch_validation'][0]['train_loss'],Train_loss_last=last['train_loss'],
 control='Conditional control skipped: no learned gain; original certified reference remains selected.',
 conclusion='Stable direct-MAE readout did not generalize; training loss fell while primary Validation worsened. Do not keep varying this head without new evidence.',
 limitations=['One seed and reused Validation; this rejects this implementation, not all possible MAE objectives.',
 'The six successful epochs include resumed epoch2 after a discarded memory-failed attempt; original sources and failure evidence remain archived.',
 'Initial teacher was trained on allTrain; no out-of-fold representation claim.',
 'No Test opened and no new uncertainty calibration claim.'],
 all_six_epoch_metrics_predictions_preserved=True,source_sha256=sha(__file__),Test='sealed',goal_73_5_confirmation_achieved=False)
write(OUT/'conclusion_mediana_directa.json',result)
# Archive metadata before removing only explicitly discarded cache/model/duplicate files.
pilot=OUT/'piloto_reloj_fino_train_s0';assert not read(pilot/'resultado.json')['passed']
cache=OUT/'datos_mediana_directa_s0';cache_meta=read(cache/'resultado.json')
targets=[pilot/'control.cbm',pilot/'cyclic.cbm',cache/'puntos_referencia.npz',cache/'resultado.json',archive/'predicciones_avance.npz']
manifest=[]
for path in targets:
 path=path.resolve();assert path.is_relative_to(OUT.resolve()) and path.is_file()
 assert path.stat().st_size<64*1024**2
 manifest.append(dict(file=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path)))
cleanup=dict(reason='Negative gate models, negative-trial frozen point cache, and epoch1archive duplicate now exactly represented in final prediction archive.',
 files=manifest,cache_metadata_preserved=cache_meta,epoch1_duplicate_all_arrays_exact_final=True,
 final_full_predictions_file=str(run/'validacion.npz'),final_full_predictions_sha256=sha(run/'validacion.npz'),
 all_originals_best_models_TransferTraj_and_all_completed_predictions_preserved=True,
 reconstruction='Frozen trainer recreates point cache automatically when its directory does not exist. Gate source recreates discarded CatBoost models.',
 removed_bytes=sum(q['bytes'] for q in manifest))
write(OUT/'limpieza_mediana_negativa.json',cleanup)
for path,item in zip(targets,manifest):
 path=path.resolve();assert path.is_relative_to(OUT.resolve()) and path.stat().st_size==item['bytes'] and sha(path)==item['sha256'];path.unlink()
assert cache.resolve().parent==OUT.resolve() and not list(cache.iterdir());cache.rmdir()
print(json.dumps(dict(best_learned_validation=best['validation'],selected_existing_reference=r['validation'],new_learned_gain=False,control='skipped per negative gate',removed_bytes=cleanup['removed_bytes'])))
