from common import *
PAIRS=[
('od_xgboost_extremos_proyectados_s0','od_xgboost_mapa2014_s0','Proyección sobre arcos frente a ajuste a nodos, mapa2014.'),
('od_xgboost_movimiento_mapmatched_s0','od_xgboost_extremos_proyectados_s0','Añadir historias por vía a las mismas rutas.'),
('od_tabm_proyectado_mm_calendario_s0_fp32_v2','od_tabm_rutas_completas_s0_fp32_v2','Paquete de proyección, historia vial y calendario; cambian varios recursos.'),
('od_mezcla_proyectada_mm_calendario_s0_fp32_v2','od_tabm_proyectado_mm_calendario_s0_fp32_v2','Distribución de hipótesis de costo frente a regresión directa, mismas entradas.'),
('od_tabm_ruta_preferida_s0_fp32_v2','od_tabm_proyectado_mm_calendario_s0_fp32_v2','Añadir ruta aprendida al mismo TabM.'),
('od_xgboost_ruta_preferida_s0','od_xgboost_movimiento_mapmatched_s0','Añadir ruta aprendida al mismo XGBoost.'),
('od_xgboost_atributos_viales_s0','od_xgboost_movimiento_mapmatched_s0','Añadir atributos físicos sobre las mismas rutas.'),
('od_mezcla_vial_continuacion_s0','od_mezcla_proyectada_mm_calendario_s0_fp32_v2','Interpolación de mediana y continuación; no aísla solo más épocas.'),
('od_mezcla_vial_mediana_s0','od_mezcla_vial_continuacion_s0','Pérdida absoluta de mediana frente al control de continuación.'),
('od_mezcla8_sin_geometria_s0_fp32_v2','od_mezcla_proyectada_mm_calendario_s0_fp32_v2','Ocho frente a cuatro componentes de duración, mismas entradas.'),
('od_mezcla8_geometria_conjunta_s0_fp32_v2','od_mezcla8_sin_geometria_s0_fp32_v2','Supervisión conjunta de forma frente a control de peso cero.'),
('od_mezcla8_pausas_condicionales_s0_fp32_v2','od_mezcla8_sin_geometria_s0_fp32_v2','Supervisión de pausas aparentes frente a control de peso cero.'),
('od_transformer_pre30_s0_fp32_v2','od_transformer_pre5_s0_fp32_v2','Presupuesto de preentrenamiento,5 vs30 épocas.'),
('od_transformer_pre30_semantica_s0_fp32_v2','od_transformer_pre5_semantica_s0_fp32_v2','Mismo contraste con contexto semántico.')]
out=[];missing=[]
for name,parent,note in PAIRS:
    if not all((RUNS/n/'resultado.json').exists() for n in (name,parent)):missing.append(name);continue
    r=json.loads((RUNS/name/'resultado.json').read_text());b=json.loads((RUNS/parent/'resultado.json').read_text());a=np.load(RUNS/name/'validacion.npz');z=np.load(RUNS/parent/'validacion.npz');assert np.array_equal(a['row'],z['row']) and np.array_equal(a['true'],z['true']);difference=np.abs(a['pred'].astype(float)-a['true'])-np.abs(z['pred'].astype(float)-z['true']);days,inv=np.unique(a['timestamp']//86400,return_inverse=True);count=np.bincount(inv);sums=np.bincount(inv,weights=difference);rng=np.random.default_rng(1729);ix=rng.integers(0,len(days),(4000,len(days)));boot=sums[ix].sum(1)/count[ix].sum(1);start=rng.integers(0,len(days),(4000,int(np.ceil(len(days)/7))));weeks=((start[:,:,None]+np.arange(7))%len(days)).reshape(4000,-1)[:,:len(days)];weekboot=sums[weeks].sum(1)/count[weeks].sum(1)
    out.append(dict(id=name,reference=parent,mae_s=r['validation']['mae_s'],reference_mae_s=b['validation']['mae_s'],delta_mae_s=float(difference.mean()),ci95_departure_day=np.quantile(boot,[.025,.975]),ci95_circular7day_blocks=np.quantile(weekboot,[.025,.975]),validation_days=len(days),note=note,record_hashes={n:sha(RUNS/n/'resultado.json') for n in (name,parent)}))
report=dict(comparisons=out,pending=missing,interpretation='Negative delta favors the new configuration. Both interval procedures are exploratory, conditional on fitted models and adaptive validation;7day circular blocks are a serial-dependence sensitivity, not a guarantee under temporal drift.',test_labels_read=False);write_json(BASE/'instrumentos/comparaciones_viales_canonicas.json',report)
lines=['# Comparaciones de la fase vial','','Todos los valores son MAE de validación en segundos. Delta negativo favorece la nueva configuración. La prueba sigue reservada.','','| Nueva configuración | Referencia | MAE | Delta | IC95 por día | IC95 bloques de 7 días |','|---|---|---:|---:|---|---|']
for r in out:lines.append('| `'+r['id']+'` | `'+r['reference']+'` | '+f"{r['mae_s']:.3f} | {r['delta_mae_s']:+.3f} | "+', '.join(f'{v:+.3f}' for v in r['ci95_departure_day'])+' | '+', '.join(f'{v:+.3f}' for v in r['ci95_circular7day_blocks'])+' |')
lines.extend(['','La validación fue consultada de forma adaptativa. Los intervalos describen diferencias condicionadas a estos modelos, sin corregir selección ni demostrar comportamiento futuro. Los bloques de siete días exploran dependencia entre días; solo hay 35 días de validación.'])
for r in out:lines.append('\n- `'+r['id']+'`: '+r['note'])
(BASE/'COMPARACIONES_VIALES.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print([(r['id'],r['delta_mae_s'],r['ci95_circular7day_blocks'].tolist()) for r in out],flush=True)
