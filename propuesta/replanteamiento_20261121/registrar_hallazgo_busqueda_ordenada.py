"""Audit completed source-coverage instruments and record the next falsifiable stage."""
from base_replanteo import *
access=deny_test()
extra=read(OUT/'cobertura_fuentes_ampliadas_Train_s0/resultado.json')
v1=read(OUT/'piloto_busqueda_ordenada_Train_s0/resultado.json')
v2=read(OUT/'piloto_busqueda_ordenada_bbox_Train_s0/resultado.json')
assert extra['complete'] and not extra['passed']
assert v2['complete'] and v2['passed'] and v2['sourceendpoint_bbox_filter_preserved']
assert v2['minimumRSS_never_increased'] and v2['original_support_never_lost'] and v2['synthetic_and_real_bruteforce_exact']
assert v1['coverage_sha256']==v2['coverage_sha256']
for name,result,key,file in [
 ('cobertura_fuentes_ampliadas_Train_s0',extra,'predictions_sha256','cobertura_consultas_Train.npz'),
 ('piloto_busqueda_ordenada_Train_s0',v1,'coverage_sha256','cobertura_consultas_Train.npz'),
 ('piloto_busqueda_ordenada_bbox_Train_s0',v2,'coverage_sha256','cobertura_consultas_Train.npz')]:
 assert sha(OUT/name/file)==result[key]
conclusion=dict(complete=True,
 additional_data_effect=dict(zero_source_before=extra['before']['zero_source_fraction'],zero_source_after=extra['after']['zero_source_fraction'],
 gained_queries=extra['gained_queries'],lost_queries=extra['lost_queries'],persist_additional_bank=False),
 ordered_search_effect=dict(before=v2['before'],after=v2['after'],within100m_gain_percentage_points=v2['within100m_gain_percentage_points'],
 bbox_filter_preserved=True,exact_same_outputs_as_initial_prototype=True,bruteforce_exact=True,perquery_old_support_preserved=True),
 interpretation='Fixed independent endpoint-neighbor truncation hides ordered matches. Parent-union discovery followed by within-parent ordered optimization corrects that geometric mechanism on512Trainqueries.',
 limitation='No new MAE result. Reused smallTraininstrument, not primaryValidation, confirmation, or proof that target73.5 is attainable.',
 new_GPS_bank_written=False,best_model_changed=False,Test='SEALED')
write(OUT/'conclusion_busqueda_ordenada_Train.json',conclusion)
plan=dict(id='busqueda_ordenada_fuentes',hypothesis='Improve actual source-time signal by removing independent endpoint-topK intersection, rather than increasing K or adding more GPS.',
 geometry_gate=dict(passed=True,result='piloto_busqueda_ordenada_bbox_Train_s0/resultado.json'),
 next_required=[
 'Implement raw5 sourceconstructor returning deterministic top64 parents/orderedindices and weights; same pointbank, sameK2048, sourceendpointbbox/span/speed filters.',
 'Use public parentboundingboxes only for mathematically safe pruning; verify exact top64 versus unpruned candidate search and independent exhaustive synthetic intervalmatching. Clock/weekend penalties are nonpositive, so spatialbounds can upper-bound weight.',
 'Exclude the whole sourceparent UTCweekfold BEFORE candidate-tree construction for every Trainquery. Arrival beforeValidation; no query N/T/path and no newqueryfields.',
 'First test a chronological Train-only timeprediction prototype without a parent pretrained on heldTrainlabels. SourceCDF/full-versus-internal summaries plus206publicfeatures, matched geometry-onlycontrol; fitparents obtain fold-excluded sources.',
 'Require at least5s gain over a fresh no-memory baseline and storedreference, and2s over sourcegeometry-onlycontrol with exploratorydayCIupper<0. Do not proceed to primaryValidation solely because geometriccoverage improved.',
 'Only if timepredictiongate passes, train the full unchanged-primary model with frozen protocol, original targets and independently reproducible raw5 inference.'
 ],
 resources='No additionalGPSbank, no all-pairs bank. Reuse241.76MiB pointbank; compactsourceindices only if justified. OneheavyGPU at a time.',
 no_sweep='Do not tune K, matchingradius, sourcecount, hiddenquery information or cohortfilters after outcomes.',
 confirmation_goal_s=73.5,goal='ACTIVE',Test='SEALED')
write(OUT/'plan_busqueda_ordenada_fuentes.json',plan)
state=read(OUT/'CONTINUIDAD_OPERATIVA.json')
state['completed_ordered_search_geometry']=conclusion
state['next_after_result']=[
 'Inspect live38228/session80329 and close boundary/densityaugmentation only after actual terminalstatus/result. Preserve all completed outcomes regardless131.06.',
 'Orderedsourcegeometrygate passed with unchangedbbox:67.7734percent to98.8281percent within100m on512Trainqueries. This is NOT a MAE result.',
 'Implement/audit deterministic top64 orderedsourceconstructor with safe parentbboxpruning; then a heldTrain timeprediction gate without pretrainedheldlabel leakage. See plan_busqueda_ordenada_fuentes.json.',
 'Do not persist expandedGPSbank: extra22.16Mpoints worsened fixedK support. Do not sweep K or sourceweights to repair a failed recipe.',
 'BestValidation129.990034 unchanged, objective73.5unmet, Testsealed; all original/TransferTraj/qualified models protected.'
]
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
p=BASE/'ARQUITECTURA.md';s=p.read_text(encoding='utf-8');marker='<!-- hallazgo_busqueda_ordenada_Train -->'
if marker not in s:
 s+='\n\n'+marker+'\n## Búsqueda conjunta de extremos: mecanismo geométrico verificado\n\n'
 s+='Se auditaron 278.941 padres adicionales anteriores al corte Train, con 22.161.411 puntos, sin guardar otro banco GPS. En 512 consultas Train, añadirlos con el mismo presupuesto de 2.048 vecinos por extremo empeoró la cobertura: se perdieron 40 consultas y se ganaron 15. Las consultas sin fuente aumentaron del 31,84 % al 36,72 %. No se amplía el banco por volumen de datos.\n\n'
 s+='El mecanismo limitante es la intersección de dos listas truncadas por separado. El nuevo instrumento utiliza esas listas para descubrir la unión de padres y busca dentro de cada uno el intervalo ordenado que minimiza la distancia a ambos extremos, con duración de fuente de 60 a 1.785 s y las mismas restricciones físicas. Una ventana móvil con respaldo exacto evita enumerar y guardar todos los pares posibles.\n\n'
 s+='Con el banco original, la proporción con una fuente a menos de 100 m de error combinado pasó de 67,7734 % a 98,8281 %. Se conservaron todas las coincidencias anteriores y ningún mínimo geométrico empeoró. La verificación independiente por enumeración coincidió; invertir el orden y consultar un caso individual dio los mismos valores. Repetir el instrumento manteniendo exactamente la máscara geográfica original produjo archivos de resultados idénticos.\n\n'
 s+='Esta evidencia justifica probar un constructor más eficiente y una predicción temporal Train sin filtración de etiquetas reservadas. No demuestra una mejora de MAE y no activa directamente otro entrenamiento primario. La siguiente etapa y su criterio de descarte están fijados en replanteamiento_20261121/plan_busqueda_ordenada_fuentes.json. El mejor MAE permanece en 129,990034 s y Test sigue cerrado.\n'
p.write_text(s,encoding='utf-8')
p=BASE/'RESUMEN_RESULTADOS.md';s=p.read_text(encoding='utf-8')
if marker not in s:
 s+='\n\n'+marker+'\n## Instrumento geométrico: búsqueda conjunta de fuentes\n\n'
 s+='En 512 consultas Train reutilizadas, la búsqueda ordenada dentro de los padres descubiertos elevó de 67,7734 % a 98,8281 % la proporción con una fuente a menos de 100 m de error combinado. Se mantuvo la máscara geográfica y se comprobó el algoritmo por enumeración independiente. No es MAE ni confirmación. Se conservan las salidas y fuentes; ningún peso del mejor modelo cambió. El aumento del banco con 22.161.411 puntos adicionales no pasó su criterio de cobertura y no se guardó otro banco GPS.\n'
p.write_text(s,encoding='utf-8')
print(json.dumps(dict(audited=True,new_MAE_claim=False,within100m_before=v2['before']['source_within100m_fraction'],within100m_after=v2['after']['source_within100m_fraction'],source_bbox_preserved=True)))
