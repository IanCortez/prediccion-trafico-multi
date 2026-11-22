from common import *
from features import block_interval
PARENTS={
'od_arbol_mae_escalado_d10_s0':'od_arbol_mae_d8_s0',
'od_residual_multiescala_s0':'od_arbol_mae_escalado_d10_s0',
'od_residual_sin_rejillas_s0':'od_residual_multiescala_s0',
'od_mas_call_mae_d10_s0':'od_arbol_mae_escalado_d10_s0',
'od_residual_preentrenado_subtrayectos_s0':'od_residual_multiescala_s0',
'od_rutas_mae_d10_s0':'od_arbol_mae_escalado_d10_s0',
'od_residual_rutas_s0':'od_residual_multiescala_s0',
'od_rutas_residuo_mae_d12_s0':'od_rutas_mae_d10_s0',
'od_contexto_rutas_mae_d10_s0':'od_rutas_mae_d10_s0',
'od_film_contexto_rutas_s0':'od_residual_rutas_s0',
'od_transformer_pre5_s0':'od_transformer_sin_pre_s0',
'od_transformer_pre5_semantica_s0':'od_transformer_pre5_s0',
'od_historial_contexto_rutas_mae_d10_s0':'od_contexto_rutas_mae_d10_s0',
'od_contexto_rutas_reciente60_s0':'od_contexto_rutas_mae_d10_s0',
'od_lightgbm_contexto_rutas_s0':'od_contexto_rutas_mae_d10_s0',
'od_xgboost_contexto_rutas_s0':'od_contexto_rutas_mae_d10_s0',
'od_movimiento_gps_contexto_rutas_d10_s0':'od_contexto_rutas_mae_d10_s0',
'od_distribucion_pre5_semantica_s0':'od_transformer_pre5_semantica_s0',
'od_transformer_geometria_aux_s0':'od_transformer_pre5_semantica_s0',
'od_estado_cache_mae_sin_inv_s0':'original_completo_pace_pos_s0',
'od_estado_gps_cache_mae_sin_inv_s0':'od_estado_cache_mae_sin_inv_s0',
'od_sufijos_adaptador_s0':'od_transformer_pre5_semantica_s0',
'od_tabm_ensemble16_s0':'od_xgboost_contexto_rutas_s0',
'od_tabm_ensemble16_s1':'od_tabm_ensemble16_s0',
'od_tabm_geometria_rutas_s0':'od_tabm_ensemble16_s0',
'od_xgboost_gps_contexto_rutas_s0':'od_xgboost_contexto_rutas_s0',
'od_xgboost_reloj_semanal_s0':'od_xgboost_contexto_rutas_s0',
'od_xgboost_geometria_rutas_s0':'od_xgboost_contexto_rutas_s0',
'od_sufijos_sin_adaptador_s0':'od_sufijos_adaptador_s0'
}
PARENTS.update({'od_tabm_rutas_completas_s0':'od_tabm_geometria_rutas_s0','od_mezcla_rutas_aux_s0':'od_tabm_rutas_completas_s0','od_mezcla_rutas_sin_aux_s0':'od_mezcla_rutas_aux_s0','od_xgboost_gps_horario_s0':'od_xgboost_contexto_rutas_s0'})
NOTES={
'od_arbol_mae_d8_s0':'MAE sin escalar converge lentamente en GPU; no constituye una referencia fuerte. Corregir escala antes de juzgar el modelo. El ajuste físico diagnóstico float32 se corrigió posteriormente.',
'od_arbol_rmse_d8_s0':'RMSE mejora MAE frente al primer intento. El ajuste físico float32 impreso no interviene en este árbol directo. El hash del código se tomó al finalizar durante una edición; las corridas siguientes conservan una copia inicial.',
'od_arbol_mae_escalado_d10_s0':'Corregir la escala permite entrenar MAE de manera efectiva. También cambió la profundidad; la diferencia no aísla un único factor.',
'od_residual_multiescala_s0':'La capacidad neuronal espacial no supera al árbol; aumenta el sobreajuste cuando continúa bajando el error de entrenamiento.',
'od_residual_sin_rejillas_s0':'Retirar las rejillas mide su aportación dentro de este modelo. Su ventaja es pequeña frente a la distancia hasta la meta.',
'od_mas_call_mae_d10_s0':'CALL_TYPE cambia el contrato de información. Se conserva como variante enriquecida y se excluye de la selección OD estricta.',
'original_diagnostico_pace_pos_s0':'480 etiquetas de otra muestra. Sirve para depurar pace/posición, pero su MAE no se compara directamente con la cohorte completa.',
'od_residual_preentrenado_subtrayectos_s0':'Los subtrayectos reutilizan etiquetas temporales de entrenamiento. El efecto favorable es modesto; no son datos sin etiquetas ni prueba de transferencia entre ciudades.',
'od_rutas_mae_d10_s0':'Longitud y tiempo libre dirigidos aportan una mejora pequeña. Se conserva el mapa con geometría y sentidos de circulación.',
'od_residual_rutas_s0':'El mapa no elimina la desventaja de este codificador neuronal frente al tabular.',
'od_rutas_residuo_mae_d12_s0':'Mayor profundidad y predicción residual no mejoran el árbol anterior; no seguir aumentando capacidad por defecto.',
'od_vecinos_historicos_s0':'La mediana ponderada de vecinos OD/tiempo pierde frente a los modelos paramétricos. La recuperación local simple no resuelve las rutas ambiguas.',
'od_contexto_rutas_mae_d10_s0':'Los conteos urbanos de ambos extremos añaden una ganancia medible y se conservan para comparadores con entradas iguales.',
'od_film_contexto_rutas_s0':'FiLM temporal y base de red no mejoran el mejor comparador. El resultado fue guardado correctamente antes de un error de serialización de consola, ya corregido.',
'od_transformer_pre5_s0':'El preentrenamiento en trayectorias completas mejora poco frente al mismo transformer desde cero. Se cuentan todas las trayectorias usadas como recursos de entrenamiento.',
'od_transformer_pre5_semantica_s0':'El texto urbano público aporta poco con esta representación. No reproduce exactamente los embeddings no especificados de TransferTraj.',
'od_historial_contexto_rutas_mae_d10_s0':'Las estadísticas históricas fuera del bloque propio evitan autoetiquetado, pero no mejoran la referencia. No se conservan por defecto.',
'od_contexto_rutas_reciente60_s0':'Dar más peso a 60 días recientes empeora la validación. La recencia uniforme no explica la brecha.',
'od_lightgbm_contexto_rutas_s0':'Otro tipo de partición de árboles no garantiza mejora; comparar costo y semillas, no solo una cifra.',
'od_xgboost_contexto_rutas_s0':'Referencia fuerte con las mismas entradas OD, mapa y contexto. La ganancia frente a CatBoost es pequeña y se interpreta con diferencias pareadas.',
'ORACULO_ruta_observada9_s0':'Control privilegiado: usa la ruta real remuestreada sin cantidad de puntos ni tiempos interiores. No es OD ni una cota de error irreducible; está prohibido para la meta.',
'original_completo_pace_pos_s0':'Comparación completa del original con pace, posición y contratos corregidos. Sirve como referencia de arquitectura; no es una réplica byte a byte de sus errores iniciales.',
'od_movimiento_gps_contexto_rutas_d10_s0':'Los tiempos de rutas usan movimiento GPS de entrenamiento excluyendo el bloque consultado. El efecto en CatBoost es mínimo; no son velocidades exactas de arcos map-matched.',
'od_distribucion_pre5_semantica_s0':'Una distribución discreta permite medir incertidumbre, pero su mediana no mejora sustancialmente el error puntual. La cobertura exploratoria no garantiza cobertura futura.',
'od_transformer_geometria_aux_s0':'Supervisar la forma de la trayectoria solo en entrenamiento aporta poco. La forma promedio puede mezclar rutas incompatibles; probar candidatos calculados.',
'od_estado_cache_mae_sin_inv_s0':'Desacoplar y reutilizar estado reduce trabajo repetido. Cambian cache, pérdida e invariancia; la comparación con el original no atribuye la diferencia a uno de ellos.',
'od_estado_gps_cache_mae_sin_inv_s0':'La velocidad local GPS semanal no supera al estado de ritmo OD en esta arquitectura. No se justifica conservarla solo porque su interpretación física sea mejor.',
'od_sufijos_adaptador_s0':'Los sufijos usan el destino verdadero y el reloj del punto actual, con adaptación de tarea. No aparece un salto de precisión; falta aislar el adaptador frente a la mezcla sin él.',
'od_tabm_ensemble16_s0':'TabM es un comparador publicado en ICLR 2025, no una aportación propia. Su mejora motiva repetir semillas antes de atribuir ventajas a más atención o transferencia.',
'od_tabm_ensemble16_s1':'Repetición de semilla del mejor comparador neuronal. Medir dispersión y ensamble simple; no elegir solo la semilla favorable.',
'od_tabm_geometria_rutas_s0':'Ablación de forma y giros de dos rutas calculadas en mapa, sin trayectoria observada de la consulta. Se usa el mismo presupuesto de etiquetas.',
'od_xgboost_gps_contexto_rutas_s0':'Comprueba si la utilidad de los perfiles GPS depende del tipo de árbol; comparar con XGBoost idéntico sin esos perfiles.',
'od_xgboost_reloj_semanal_s0':'Se retiran mes y ciclos anuales porque no hay un año completo en entrenamiento. El reloj semanal y el resto de las entradas se conservan.',
'od_xgboost_geometria_rutas_s0':'Mide geometría de candidatos calculados con el mismo XGBoost de referencia.',
'od_sufijos_sin_adaptador_s0':'Control de mezcla de tiempos restantes y completos sin prompt privado: aísla el adaptador dentro del mismo entrenamiento.'
}
NOTES.update({'od_tabm_rutas_completas_s0':'Control directo con geometría estática, costos GPS y ruta dinámica. El conjunto de características coincide con la mezcla; esta comparación con el TabM anterior añade varios recursos a la vez.','od_mezcla_rutas_aux_s0':'Tres rutas calculadas y un componente residual modelan una distribución de duración. La predicción es la mediana de la mezcla. Comparar contra el predictor directo y contra la ablación sin forma auxiliar; no atribuir una diferencia al adaptador o a la distribución por separado.','od_mezcla_rutas_sin_aux_s0':'Ablación de supervisión de forma observada, limitada al entrenamiento. Mismas características OD y distribución de duración.','od_xgboost_gps_horario_s0':'Estado GPS de 168 horas semanales y 500 m con dos bloques excluyentes. Frente al perfil previo cambian también resolución espacial, direcciones y folds; esta corrida no aísla solo el ancho temporal.'})
PARENTS.update({'od_xgboost_mapa2014_s0':'od_xgboost_contexto_rutas_s0','od_xgboost_mapa2026_s0':'od_xgboost_contexto_rutas_s0','od_xgboost_calendario_s0':'od_xgboost_contexto_rutas_s0','od_xgboost_extremos_proyectados_s0':'od_xgboost_mapa2014_s0','od_xgboost_movimiento_mapmatched_s0':'od_xgboost_extremos_proyectados_s0','od_tabm_geometria_rutas_s1':'od_tabm_geometria_rutas_s0'})
NOTES.update({'od_xgboost_mapa2014_s0':'Se añade el archivo vial de enero de 2014 con rutas y forma. El estado histórico del mapa no garantiza más precisión: comprobar cobertura y comparador 2026 con extracción idéntica.','od_xgboost_mapa2026_s0':'Extracción anual moderna con las mismas reglas que 2014. Permite distinguir fecha del mapa y cobertura de la caché original.','od_xgboost_calendario_s0':'Calendario conocido antes de la salida: indicadores de días especiales, víspera, día posterior y puente. Reglas históricas 2013-2014 verificadas; no observa resultados de viajes futuros.','od_xgboost_extremos_proyectados_s0':'Se proyectan O/D en arcos y se consideran cuatro candidatos por extremo. Dos rutas estáticas con geometría válida, sin usar velocidades GPS para elegirlas. Comparar con el mismo mapa 2014 ajustado a nodos.','od_xgboost_movimiento_mapmatched_s0':'Añade tiempos y soporte por vía/hora a las mismas rutas proyectadas. El GPS es exclusivamente de entrenamiento y excluye el bloque propio en todos los niveles. Esta comparación aísla las nuevas estadísticas históricas.','od_tabm_geometria_rutas_s1':'Repetición independiente de la variante con geometría de rutas; evaluar dispersión y no seleccionar solamente la semilla favorable.'})
PARENTS.update({'od_tabm_proyectado_mm_calendario_s0':'od_tabm_rutas_completas_s0','od_mezcla_proyectada_mm_calendario_s0':'od_tabm_proyectado_mm_calendario_s0','od_mezcla_rutas_sin_aux_s1':'od_mezcla_rutas_sin_aux_s0','od_xgboost_ruta_preferida_s0':'od_xgboost_movimiento_mapmatched_s0','od_tabm_ruta_preferida_s0':'od_tabm_proyectado_mm_calendario_s0'})
NOTES.update({'od_tabm_proyectado_mm_calendario_s0':'Control directo con proyección de extremos, mapa 2014, historial por vía map-matched y calendario. Frente al control legado cambian varios recursos; no atribuir la mejora a uno solo.','od_mezcla_proyectada_mm_calendario_s0':'Mismas entradas que TabM directo, distribuciones de duración y mediana de mezcla. Dos candidatos de ruta, una hipótesis de tiempo libre y un componente residual; los cuatro componentes no son cuatro rutas distintas ni sus pesos probabilidades de ruta verificadas.','od_mezcla_rutas_sin_aux_s1':'Segunda semilla de la mezcla sin forma auxiliar, conservando configuración y entradas. Examinar dispersión y promedio fijo, sin seleccionar únicamente la semilla favorable.','od_xgboost_ruta_preferida_s0':'Añade forma, costos históricos y soporte de una ruta de giros aprendidos fuera del bloque propio. La exactitud local con trayectoria prefijada no garantiza calidad de ruta OD ni menor MAE.','od_tabm_ruta_preferida_s0':'Añade las mismas características de ruta aprendida al TabM con proyección, historial vial y calendario. Comparación de representación con estimador y presupuesto fijos.'})

PARENTS.update({'od_mezcla_vial_continuacion_s0':'od_mezcla_proyectada_mm_calendario_s0_fp32_v2','od_mezcla_vial_mediana_s0':'od_mezcla_vial_continuacion_s0'})
NOTES.update({'od_mezcla_vial_continuacion_s0':'Control de entrenamiento adicional desde el checkpoint vial, mediana interpolada e inferencia float32. Frente al padre cambian interpolación y continuación; no atribuir su diferencia solo a más épocas.','od_mezcla_vial_mediana_s0':'Misma inicialización, datos, semilla, mediana interpolada y presupuesto que el control de continuación. Se añade pérdida absoluta de mediana con peso fijo3, manteniendo NLL y RPS. Contraste exploratorio sin etiquetas de prueba.'})

PARENTS.update({'od_xgboost_atributos_viales_s0':'od_xgboost_movimiento_mapmatched_s0','od_tabm_atributos_viales_s0':'od_tabm_proyectado_mm_calendario_s0'})
NOTES.update({'od_xgboost_atributos_viales_s0':'Añade atributos del mapa 2014 sobre las mismas dos rutas: cruces, controles etiquetados, composición vial, carriles conocidos y variación de límites de velocidad. Missing OSM tags no equivalen a ausencia real. Comparador, datos y resto de entradas fijos.','od_tabm_atributos_viales_s0':'Mismos atributos estáticos añadidos al TabM vial con calendario. El control sigue siendo el TabM de 216 características; sin ruta aprendida ni cambios en etiquetas.'})

PARENTS.update({'od_transformer_pre30_s0':'od_transformer_pre5_s0','od_transformer_pre30_semantica_s0':'od_transformer_pre5_semantica_s0'})
NOTES.update({'od_transformer_pre30_s0':'Aumenta el preentrenamiento de5 a30 épocas continuando el mismo estado de optimizador y generador. Misma arquitectura, cohorte y ajuste OD. No es reproducción exacta de TransferTraj ni transferencia entre ciudades.','od_transformer_pre30_semantica_s0':'Mismo contraste de presupuesto, conservando el contexto semántico y el ajuste OD del control de cinco épocas.'})

PARENTS.update({'od_mezcla8_sin_geometria_s0':'od_mezcla_proyectada_mm_calendario_s0','od_mezcla8_geometria_conjunta_s0':'od_mezcla8_sin_geometria_s0'})
NOTES.update({'od_mezcla8_sin_geometria_s0':'Control con ocho componentes y las mismas216 características OD de la mezcla vial. Conserva la arquitectura del experimento conjunto con peso geométrico cero.','od_mezcla8_geometria_conjunta_s0':'Misma arquitectura y presupuesto; añade NLL condicional de forma observada solo de entrenamiento, con peso fijo0.2 y normalización por14 posterior a marginalizar. Las medias geométricas latentes no garantizan rutas viales válidas ni etiquetas de componentes identificables.'})

PARENTS['od_mezcla8_pausas_condicionales_s0']='od_mezcla8_sin_geometria_s0'
NOTES['od_mezcla8_pausas_condicionales_s0']='Añade NLL beta-binomial de intervalos GPS de bajo desplazamiento a la misma arquitectura del control sin auxiliares. El conteo y n=T/15 solo aparecen como etiquetas en la pérdida de entrenamiento. No se interpretan como estacionamiento anotado ni se filtran consultas por pausas observadas.'

def eligible(r):
    return r.get('train_n')==1023529 and r.get('cohort','primary')=='primary' and not r.get('config',{}).get('enriched',False)
def read_records():
    records=[]
    for p in RUNS.glob('*/resultado.json'):
        try:records.append(json.loads(p.read_text(encoding='utf-8-sig')))
        except json.JSONDecodeError:continue
    return records

def report_iteration(r):
    result=dict(id=r['id'],validation=r['validation'],eligible_primary=eligible(r),note=NOTES.get(r['id'],r.get('conclusion','')),test_labels_read=False)
    parent=PARENTS.get(r['id'],r.get('trained_model_source'));records={q['id']:q for q in read_records()}
    if parent in records:
        a=np.load(RUNS/r['id']/'validacion.npz');b=np.load(RUNS/parent/'validacion.npz')
        if len(a['row'])==len(b['row']) and np.array_equal(a['row'],b['row']):
            difference=np.abs(a['pred']-a['true'])-np.abs(b['pred']-b['true'])

            # block_interval takes absolute errors; signed paired deltas need their own bootstrap.
            _,inv=np.unique(a['timestamp']//86400,return_inverse=True);count=np.bincount(inv);sums=np.bincount(inv,weights=difference)
            ix=np.random.default_rng(1729).integers(0,len(count),(4000,len(count)));boot=sums[ix].sum(1)/count[ix].sum(1)
            result['comparison']=dict(reference=parent,delta_mae_s=float(difference.mean()),ci95_s=np.quantile(boot,[.025,.975]).tolist(),interpretation='Negative favors this iteration; exploratory after adaptive validation selection, conditional on fitted models.')
    result['distance_to_target_s']=float(r['validation']['mae_s']-73.5)
    result['decision']='Meta no alcanzada; conservar prueba reservada.' if r['validation']['mae_s']>=73.5 else 'Candidato de validación: requiere semillas, contratos y confirmación congelada.'
    write_json(RUNS/r['id']/'evaluacion_critica.json',result)
    return result




