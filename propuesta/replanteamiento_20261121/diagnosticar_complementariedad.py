"""Exploratory error complementarity; oracle values are explicitly ineligible predictors."""
from base_replanteo import *
access=deny_test();ref=np.load(BEST/'validacion.npz');y=ref['true'].astype(float);base=ref['pred'].astype(float);rows=ref['row'];records={};preds=[base];names=['retained_best']
for name in ['descompuesto_aux1_s0','descompuesto_aux0_s0','memoria_cdf_s0','memoria_mse_s0','base_oof_estatica_s0','espacial_semimarkov_s0_v2']:
 p=OUT/name/'validacion.npz'
 if not p.exists() or not (p.parent/'resultado.json').exists():continue
 z=np.load(p);assert np.array_equal(z['row'],rows) and np.array_equal(z['true'],y);pred=z['pred'].astype(float);e=np.abs(y-pred);eb=np.abs(y-base);delta=pred-base;records[name]=dict(validation=metric(y,pred),paired_vs_retained=paired(y,pred,base,ref['timestamp']),mean_absolute_prediction_difference_s=float(np.abs(delta).mean()),improves_over_retained_fraction=float((e<eb).mean()),two_model_ORACLE_min_absolute_error_s=float(np.minimum(e,eb).mean()),ORACLE_is_not_a_valid_model=True);preds.append(pred);names.append(name)
errors=np.abs(np.stack(preds)-y[None,:]);report=dict(scope='EXPLORATORY reused full Validation; no Test',models=names,comparisons=records,all_models_ORACLE_min_absolute_error_s=float(errors.min(0).mean()),oracle_warning='Chooses with the unknown true duration of each query. Not usable inference, not achieved MAE, not an information-theoretic lower bound. Its sole purpose is to measure complementarity before designing a selector.',Test_access=False)
write(OUT/'diagnostico_complementariedad.json',report)
# One known duplicate can be removed only after verifying exact equality to final OOF asset.
p=OUT/'base_oof_estatica_s0/oof_progreso.npz';removed=[]
if p.exists():
 final=np.load(OUT/'base_oof_estatica_s0/train_oof.npz');inter=np.load(p);assert all(np.array_equal(inter[k],final[k]) for k in ['row','fold']);assert np.array_equal(inter['pred'],final['pred_oof']);inter.close();final.close();p=p.resolve();assert p.parent==(OUT/'base_oof_estatica_s0').resolve() and p.name=='oof_progreso.npz';record=dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size,reason='Verified exact duplicate of final OOF predictions');p.unlink();removed.append(record)
if removed:write(OUT/'limpieza_intermediarios.json',dict(removed=removed,original_data_untouched=True,retained_best_untouched=True,TransferTraj_untouched=True))
print(json.dumps(dict(comparisons={k:dict(mae_s=r['validation']['mae_s'],prediction_difference_s=r['mean_absolute_prediction_difference_s'],two_model_ORACLE_s=r['two_model_ORACLE_min_absolute_error_s']) for k,r in records.items()},all_model_ORACLE_s=report['all_models_ORACLE_min_absolute_error_s'],removed_bytes=sum(r['bytes'] for r in removed))))

