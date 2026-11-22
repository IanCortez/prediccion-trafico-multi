from base_replanteo import *
d=read(OUT/'diagnostico_riesgos_temporales.json')
a=read(OUT/'prior_eventos/resultado.json')
v=read(OUT/'verificacion_prefijo_historia.json')
assert d['passed'] and v['passed']
new_assets=sum(z['bytes'] for z in a['outputs'].values())
record=dict(status='Native GPU training continues; no native full Validation result yet',positive_local_evidence=dict(held_Train_points=d['scores']['cell_heading_destination_clock_age']['held_points'],clock_effect=d['clock_effect'],age_effect=d['age_effect'],scope='Historical context likelihood; not OD MAE and not complementarity to a neural representation'),asset_verification=dict(bytes=new_assets,held_instrument_reproduced=a['held_instrument_reproduced'],fold_exclusion=read(OUT/'verificacion_historia_eventos.json'),model=v,recovered_preparation='Wrong shape assertion corrected; original count file unchanged; see prior_eventos/fallo_preparacion.json'),next_gate=dict(script='piloto_historia_condicional.py',requires='Freeze native model and complete its full127941 result first',fit='25000Train trips; parent representation frozen; new3-event heads fitted on Train weekfolds1..4, assessed on fold0. Each count lookup excludes its own fold.',matched_control='New linear head without empirical history; same frozen hidden representation and support',original_control='Original frozen native stay/stop event probabilities',criterion='History must reduce held NLL by>.001nats versus BOTH controls and have day-bootstrap upper95%<0',scope='Only newly fitted heads held out, parent representation saw Train; not whole-model generalization',if_positive='Warm-start6epochs new HistoricalPrefixArrival with controlled head; full Validation required; improvement alone is not causal attribution to prior because of additional training',if_negative='Do not run temporal-prior GPU trial; pursue a distinct mechanism, not blend or smoothing searches'),goal='active',final_confirmation_s=73.5,preservation_threshold_s=131.06,Test='sealed')
write(OUT/'plan_control_historia_condicional.json',record)
c=read(OUT/'CONTINUIDAD_OPERATIVA.json')
c['active_unified_session']=9112
c['history_pipeline_session']=46174
c['history_pipeline_supervisor_pid']=35436
c['history_pipeline_state_file']='pipeline_historia_estado.json'
c['historical_prior_diagnostic_and_control_plan']=record
hist=read(OUT/'prefijo_nativo_s0/historia.json')
c['latest_native_checkpoint_selection_only']=hist[-1]
c['next_after_native']=['Wait real PID20612 and supervisor33120; held process handles are in pipeline supervisor35436/session46174','Read complete Native result and preservation manifest; alpha0 is no learned gain','CPU conditional head pilot runs automatically only after complete frozen parent result','Only if conditional pilot beats both controls will prefijo_historia_s0 start; otherwise goal stays active and a different mechanism is needed','No duplicate GPU jobs, no edits to running sources, no Test access']
write(OUT/'CONTINUIDAD_OPERATIVA.json',c)
p=OUT/'README.md'
s=p.read_text(encoding='utf-8-sig')
marker='<!-- historia_temporal_controlada -->'
if marker not in s:
    s+='\n'+marker+'\n## Diagnóstico temporal y siguiente control\n\nCon 9.198.270 decisiones locales reservadas de Train, añadir horario UTC de entrada a zona redujo NLL de 0,558328 a 0,541826. Diferencia −0,016501, IC95 % por día [−0,018592; −0,014138]. La edad de permanencia también aporta respecto a una probabilidad constante. Esto no es un resultado de MAE de viaje.\n\nLos conteos y probabilidades por contexto ocupan 18.461.049 bytes, sin bancos por consulta. Se verificó que excluir una semana completa deja invariantes todos los niveles de respaldo. Una recuperación corrigió la comprobación de dimensiones sin recalcular ni cambiar los conteos; la nueva implementación reproduce el diagnóstico anterior con diferencia inferior a 0,000001 nats.\n\nEl siguiente paso condicionado compara cabezas equivalentes con y sin historial sobre la misma representación neuronal congelada. Se exige beneficio frente al control y frente al modelo neuronal original antes de iniciar otra GPU. Las semanas reservadas solo validan las nuevas cabezas: la representación ya vio Train. Ver plan_control_historia_condicional.json y pipeline_historia_estado.json. El entrenamiento temporal preparado no se ejecuta si falla ese control.\n'
    p.write_text(s,encoding='utf-8')
p=BASE/'RESUMEN_RESULTADOS.md'
s=p.read_text(encoding='utf-8-sig')
marker='<!-- diagnostico_historia_Train -->'
if marker not in s:
    s+='\n'+marker+'\nEl diagnóstico de historial temporal encontró una mejora local en 9.198.270 decisiones reservadas de Train: NLL 0,558328 → 0,541826, con intervalo por día de la diferencia por debajo de cero. Todavía no demuestra mejora del MAE del viaje ni información adicional frente al modelo neuronal. Antes de otra corrida GPU se compararán cabezas equivalentes con/sin historial y el modelo neuronal original, sobre el padre congelado y sin usar GPS de Validación o Test. El estado vigente sigue en los registros de ejecución.\n'
    p.write_text(s,encoding='utf-8')
print(json.dumps(dict(prior_assets_MiB=new_assets/1024**2,next_gate='Matched conditional CPU heads after full native result',goal='active',Test='sealed')))
