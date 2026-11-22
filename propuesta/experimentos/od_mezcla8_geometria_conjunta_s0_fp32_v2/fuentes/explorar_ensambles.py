from common import *
from features import *
# Predefined equal-weight families; no learned weights fitted on validation.
families={
'TabM_semillas01':(['od_tabm_ensemble16_s0','od_tabm_ensemble16_s1'],'mean'),
'TabM_y_XGBoost':(['od_tabm_ensemble16_s0','od_xgboost_gps_contexto_rutas_s0'],'mean'),
'TabM01_y_XGBoost':(['od_tabm_ensemble16_s0','od_tabm_ensemble16_s1','od_xgboost_gps_contexto_rutas_s0'],'mean'),
'TabM01_y_XGBoost_mediana':(['od_tabm_ensemble16_s0','od_tabm_ensemble16_s1','od_xgboost_gps_contexto_rutas_s0'],'median'),
'TabM_geometria_y_base':(['od_tabm_ensemble16_s0','od_tabm_geometria_rutas_s0'],'mean'),
'TabM_geometria_y_XGBoost':(['od_tabm_geometria_rutas_s0','od_xgboost_gps_contexto_rutas_s0'],'mean')
}
families['TabM_geometria_y_XGBoost_horario']=(['od_tabm_geometria_rutas_s0','od_xgboost_gps_horario_s0'],'mean')

families.update({
'TabM_geometria_semillas01':(['od_tabm_geometria_rutas_s0','od_tabm_geometria_rutas_s1'],'mean'),
'Mezcla_legado_semillas01':(['od_mezcla_rutas_sin_aux_s0','od_mezcla_rutas_sin_aux_s1'],'mean'),
'Mezcla_y_TabM_legado':(['od_mezcla_rutas_sin_aux_s0','od_tabm_rutas_completas_s0'],'mean'),
'Mezcla_legado_y_XGBoost_vial':(['od_mezcla_rutas_sin_aux_s0','od_xgboost_movimiento_mapmatched_s0'],'mean'),
'Mezcla_y_TabM_vial':(['od_mezcla_proyectada_mm_calendario_s0','od_tabm_proyectado_mm_calendario_s0'],'mean'),
'Mezcla_TabM_XGBoost_vial':(['od_mezcla_proyectada_mm_calendario_s0','od_tabm_proyectado_mm_calendario_s0','od_xgboost_movimiento_mapmatched_s0'],'mean'),
'Mezclas_dos_representaciones':(['od_mezcla_rutas_sin_aux_s0','od_mezcla_proyectada_mm_calendario_s0'],'mean'),
'Dos_mezclas_y_TabM_XGBoost_vial':(['od_mezcla_rutas_sin_aux_s0','od_mezcla_proyectada_mm_calendario_s0','od_tabm_proyectado_mm_calendario_s0','od_xgboost_movimiento_mapmatched_s0'],'mean')})
families['Mediana_dirigida_y_TabM_vial']=(['od_mezcla_vial_mediana_s0','od_tabm_proyectado_mm_calendario_s0'],'mean')
families['Mezcla8_conjunta_y_Mezcla4_vial']=(['od_mezcla8_geometria_conjunta_s0','od_mezcla_proyectada_mm_calendario_s0'],'mean')
families['Mezcla8_control_y_conjunta']=(['od_mezcla8_sin_geometria_s0','od_mezcla8_geometria_conjunta_s0'],'mean')
out=[];folder=BASE/'instrumentos/ensambles_validacion';folder.mkdir(exist_ok=True)
for name,(members,aggregation) in families.items():
    members=[m+'_fp32_v2' if (RUNS/(m+'_fp32_v2')/'resultado.json').exists() else m for m in members]
    if not all((RUNS/m/'resultado.json').exists() for m in members):continue
    arrays=[];hashes={};first=None
    for member in members:
        z=dict(np.load(RUNS/member/'validacion.npz'))
        if first is None:first=z
        else:assert np.array_equal(z['row'],first['row']) and np.array_equal(z['true'],first['true'])
        arrays.append(z['pred']);hashes[member]=sha(RUNS/member/'validacion.npz')
    pred=(np.mean if aggregation=='mean' else np.median)(np.stack(arrays).astype(float),axis=0)
    rec=dict(id=name,members=members,aggregation=aggregation,validation=metrics(first['true'],pred),interval=block_interval(first['true'],pred,first['timestamp']),prediction_hashes=hashes,test_labels_read=False,weights_fitted_to_validation=False,caveat='Exploratory comparison of predefined families on repeatedly used validation. Independent confirmation remains sealed.')
    out.append(rec);np.savez_compressed(folder/(name+'.npz'),row=first['row'],pred=pred,true=first['true'],timestamp=first['timestamp'])
out.sort(key=lambda r:r['validation']['mae_s']);write_json(BASE/'instrumentos/ensambles_validacion.json',out)
seeds=[json.loads((RUNS/name/'resultado.json').read_text(encoding='utf-8-sig'))['validation']['mae_s'] for name in ('od_tabm_ensemble16_s0','od_tabm_ensemble16_s1')]
write_json(BASE/'instrumentos/semillas_tabm.json',dict(individual_mae_s=seeds,mean_s=float(np.mean(seeds)),sample_std_s=float(np.std(seeds,ddof=1)),n=2,caveat='Only two seeds; dispersion estimate is uncertain, not a confidence interval or confirmation.'))
print([(r['id'],r['validation']['mae_s']) for r in out],flush=True)

