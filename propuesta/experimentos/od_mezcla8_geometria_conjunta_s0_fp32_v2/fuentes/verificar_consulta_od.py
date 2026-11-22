from common import *
from consulta_od import *
from predecir_od import predict_new_od
from inferencia import feature_input
import torch,gc
va=dict(np.load(DATA/'validation_x.npz'));routes=np.load(DATA/'validation_rutas.npz')['X'];rng=np.random.default_rng(728);ix=np.unique(np.r_[rng.choice(len(va['row']),60,replace=False),np.flatnonzero(routes[:,6]==0)[:5]])
columns=('o_lon','o_lat','d_lon','d_lat','timestamp');items=[{key:va[key][i].item() for key in columns} for i in ix]
checks=[]
for name in ('od_tabm_ensemble16_s0_fp32_v2','od_tabm_geometria_rutas_s0_fp32_v2','od_tabm_proyectado_mm_calendario_s0_fp32_v2','od_mezcla_proyectada_mm_calendario_s0_fp32_v2','od_tabm_ruta_preferida_s0_fp32_v2','od_mezcla_vial_continuacion_s0','od_mezcla_vial_mediana_s0','od_mezcla8_sin_geometria_s0_fp32_v2','od_mezcla8_geometria_conjunta_s0_fp32_v2','od_transformer_pre30_s0_fp32_v2','od_transformer_pre30_semantica_s0_fp32_v2','od_mezcla8_pausas_condicionales_s0_fp32_v2'):
    if not (RUNS/name/'resultado.json').exists():continue
    record=json.loads((RUNS/name/'resultado.json').read_text(encoding='utf-8-sig'));pred,X,names=predict_new_od(RUNS/name,items);_,stored,stored_names=feature_input('validation',record)
    assert names==stored_names;feature_error=float(np.max(np.abs(X-stored[ix])));z=np.load(RUNS/name/'validacion.npz');prediction_error=float(np.max(np.abs(pred-z['pred'][ix])))
    assert feature_error<1e-4 and prediction_error<.1,(name,feature_error,prediction_error)
    checks.append(dict(id=name,n=len(ix),max_feature_change=feature_error,max_prediction_change_s=prediction_error));gc.collect();torch.cuda.empty_cache()
try:od_records([dict(items[0],duration=100)])
except AssertionError:pass
else:raise RuntimeError('Forbidden duration accepted')
write_json(BASE/'instrumentos/reconstruccion_desde_od.json',dict(passed=True,models=checks,forbidden_field_rejected=True,derived_query_inputs_only=columns,query_labels_used=False,test_labels_read=False,method='Fresh graph routing, static grid lookup, projected candidate construction and training-only movement history from OD/departure. Compared only afterwards with precomputed validation features/predictions.'))
write_json(BASE/'instrumentos/ejemplo_consulta_od.json',items[:3]);print(checks,flush=True)


