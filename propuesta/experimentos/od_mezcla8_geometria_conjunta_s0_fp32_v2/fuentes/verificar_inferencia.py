from common import *
from inferencia import *
import torch,gc,argparse
p=argparse.ArgumentParser();p.add_argument('--ids',nargs='+');a=p.parse_args()
names=a.ids or ['od_xgboost_calendario_s0','od_xgboost_mapa2014_s0','od_xgboost_mapa2026_s0','od_xgboost_extremos_proyectados_s0','od_xgboost_movimiento_mapmatched_s0','od_xgboost_ruta_preferida_s0','od_tabm_geometria_rutas_s1_fp32_v2','od_tabm_proyectado_mm_calendario_s0_fp32_v2','od_mezcla_rutas_sin_aux_s0_fp32_v2','od_mezcla_proyectada_mm_calendario_s0_fp32_v2','od_mezcla_rutas_sin_aux_s1_fp32_v2','od_tabm_ruta_preferida_s0_fp32_v2','od_mezcla_vial_continuacion_s0','od_mezcla_vial_mediana_s0','od_xgboost_atributos_viales_s0','od_transformer_pre30_s0','od_transformer_pre30_semantica_s0','od_mezcla8_sin_geometria_s0_fp32_v2','od_mezcla8_geometria_conjunta_s0_fp32_v2','od_transformer_pre30_s0_fp32_v2','od_transformer_pre30_semantica_s0_fp32_v2','od_mezcla8_pausas_condicionales_s0_fp32_v2']
path=BASE/'instrumentos/inferencia_reproducible.json';previous=json.loads(path.read_text()) if path.exists() else {};out={r['id']:r for r in previous.get('models',[])};tested=[];skipped=[]
for name in names:
    if not (RUNS/name/'resultado.json').exists():
        if a.ids:raise FileNotFoundError(name)
        skipped.append(name);continue
    d,pred=predict_run(RUNS/name,'validation');z=np.load(RUNS/name/'validacion.npz');assert np.array_equal(d['row'],z['row']);delta=float(np.max(np.abs(pred-z['pred'])));assert delta<.05,(name,delta);rec=dict(id=name,max_prediction_change_s=delta,n=len(pred),verified_unix=time.time(),record_sha256=sha(RUNS/name/'resultado.json'),inference_code_sha256=sha(BASE/'inferencia.py'));out[name]=rec;tested.append(name);print(rec,flush=True);gc.collect();torch.cuda.empty_cache()
try:
    predict_run(RUNS/'ORACULO_ruta_observada9_s0','validation');raise RuntimeError('Oracle was not blocked')
except AssertionError as error:assert 'Privileged' in str(error)
write_json(path,dict(models=list(out.values()),models_tested_this_execution=tested,skipped_unfinished=skipped,previously_recorded_models_retained=True,oracle_blocked=True,test_labels_read=False,passed=True))
