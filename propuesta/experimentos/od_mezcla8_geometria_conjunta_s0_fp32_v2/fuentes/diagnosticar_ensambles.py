from common import *
from features import *
from evaluar_iteracion import eligible,read_records
families={
'best_three_finished_before_diagnostic':['od_tabm_ensemble16_s0','od_tabm_ensemble16_s1','od_xgboost_gps_contexto_rutas_s0'],
'diverse_representations':['od_tabm_ensemble16_s0','od_xgboost_gps_contexto_rutas_s0','od_contexto_rutas_mae_d10_s0','od_residual_preentrenado_subtrayectos_s0','od_film_contexto_rutas_s0','original_completo_pace_pos_s0','od_transformer_pre5_semantica_s0','od_distribucion_pre5_semantica_s0']
}
results=[]
for name,members in families.items():
    z=[dict(np.load(RUNS/m/'validacion.npz')) for m in members];base=z[0]
    assert all(np.array_equal(q['row'],base['row']) and np.array_equal(q['true'],base['true']) for q in z)
    predictions=np.stack([q['pred'] for q in z]).astype(float);y=base['true'].astype(float);lo=predictions.min(0);hi=predictions.max(0)
    bound=np.maximum(np.maximum(lo-y,y-hi),0);oracle=np.abs(predictions-y).min(0)
    rng=np.random.default_rng(561)
    for _ in range(30):
        weights=rng.dirichlet(np.ones(len(members)),size=len(y)).T
        blended=(predictions*weights).sum(0)
        assert np.all(np.abs(blended-y)>=bound-1e-8)
    result=dict(family=name,members=members,n=len(y),convex_hull_mae_lower_bound_s=float(bound.mean()),best_member_oracle_mae_s=float(oracle.mean()),truth_inside_prediction_range_fraction=float(np.mean((y>=lo)&(y<=hi))),any_convex_mixture_can_reach73_5_on_this_validation=bool(bound.mean()<73.5),definition='For each validation row, distance of true duration to the closed interval between minimum and maximum fixed predictions. This is an exact lower bound for arbitrary nonnegative weights summing to 1, even label-informed weights.',limitation='Bound applies only to these fixed predictions on this validation cohort. It is not a Bayes-risk bound, not a held-out result, and does not constrain new predictors or blends with negative weights/residual corrections.',test_labels_read=False)
    results.append(result)
write_json(BASE/'instrumentos/limite_ensambles_convexos.json',results)
print([(r['family'],r['convex_hull_mae_lower_bound_s'],r['best_member_oracle_mae_s']) for r in results],flush=True)
