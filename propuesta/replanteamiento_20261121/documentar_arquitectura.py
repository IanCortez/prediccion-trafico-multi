from base_replanteo import *
p=OUT/'concluir_descomposicion.py';s=p.read_text(encoding='utf-8-sig').replace("print(json.dumps({k:report[k] for k in ['supervised_proxy','same_architecture_without_proxy','supervised_minus_control','control_minus_retained']}))","print(json.dumps({k:report[k] for k in ['supervised_proxy','same_architecture_without_proxy','supervised_minus_control','control_minus_retained']},default=lambda v:v.tolist()))");p.write_text(s,encoding='utf-8')
p=OUT/'actualizar_estado.py';s=p.read_text(encoding='utf-8-sig');s=s.replace('el control sin esa supervisión está en ejecución.','el control sin esa supervisión terminó en 131,4176 s y el supervisado en 131,6273 s. El incremento de error supervisado es 0,2097 s, IC95 % exploratorio por día [0,0442; 0,3703].');s=s.replace("current_best=dict(id=BEST.name,mae_s=131.25671207822356)","candidate_preservation_threshold_s=131.06,current_best=dict(id=BEST.name,mae_s=131.25671207822356)");p.write_text(s,encoding='utf-8')
p=BASE/'RESUMEN_RESULTADOS.md';s=p.read_text(encoding='utf-8-sig');mark='<!-- replanteamiento_activo_20261121 -->';assert mark in s;s=s.split(mark)[0]+mark+'''
## Replanteamiento continuo activo

El objetivo de confirmación <73,5 s permanece activo. Test sigue cerrado. La mejor referencia conservada continúa en 131,256712 s. Por indicación del usuario, los nuevos candidatos con MAE de Validación completa <131,06 s conservarán pesos, métricas, predicciones y reproducción; se protegen también el modelo anterior y todos los resultados TransferTraj.

| Mecanismo evaluado | MAE en los mismos 127.941 viajes |
|---|---:|
| Descomposición movimiento/bajo desplazamiento supervisado | 131,627313 s |
| Control con la misma descomposición, sin supervisión del conteo | 131,417567 s |
| Memoria de etiquetas con métrica entrenable, distribución | 137,431590 s |
| Memoria de etiquetas con métrica entrenable, media/MSE | 144,303155 s |
| Proceso espacial semi-Markov sin dirección | 138,223439 s |

No mejoran la referencia. La supervisión de bajo desplazamiento empeora 0,2097 s respecto a su control en esta configuración (IC95 % exploratorio por día: 0,0442 a 0,3703 s); no se generaliza ese resultado a todas las formas de modelar paradas.

Siguen en ejecución una memoria de residuos con predicciones fuera de cada bloque y un proceso espacial cuyo estado conserva dirección de llegada. Este último se justifica con semanas reservadas de Train: el acierto de siguiente zona sube de 72,75 % a 76,37 %. Es evidencia sobre transiciones, no una mejora demostrada de duración.

Se conservan predicciones y fuentes de ensayos descartados, retirando sus pesos. No se generan bancos CDF por viaje en disco. [Propuesta, conclusiones y ejecución](replanteamiento_20261121/README.md); [política de conservación](replanteamiento_20261121/politica_conservacion.json).
''';p.write_text(s,encoding='utf-8')
write(OUT/'arquitectura_activa.json',dict(goal='Same primary OD benchmark; confirmation MAE <73.5s',no_query_GPS_or_duration_or_point_count=True,arms=[
 dict(mechanism='Generative additive intervals',math='p(T|OD,t)=sum_style,move,lowmove p(style|OD,t)*p(move|style,OD,t)*p(lowmove|style,OD,t)*I[T=15*(move+lowmove)]',intervention='Conditional NLL of Train low-displacement count at known Train total duration',control='Samearchitecture with auxiliary weight0',conclusion='Does not improve; do not expand unchanged mechanism'),
 dict(mechanism='Learned historical memory',math='Original p(T|q)=softmax_negative_metric(q,Trainkeys) times label histograms; revised prediction=g_CF(q)+sigmoid(gate(q))*weighted residuals',intervention='Residuals from strict blocked cross-fit; candidates and query within same excluded Train fold',validation='No Validation labels in bank; exact all-bank streaming inference',status='training residual version'),
 dict(mechanism='OD-conditioned spatial semi-Markov',math='p(T|q)=sum over all hypothetical state paths and positive residence durations, absorbed at destination; fixed120-bin horizon',initial_state='Unknown incoming direction at origin; no true query prefix',state_revision='Cell×4 incoming cardinal headings, plus one query-specific unknown-heading initial state',test_evidence='Analytical mass/time/loop/gradient tests, full-model probability normalization and query permutation',status='direction-aware model training',antecedent='UNITE KDD2026; our model is an adaptation with exact finite-horizon marginalization, not its GRU architecture or an exact replication')
 ],next_decisions=[
 dict(hypothesis='Dependence of waiting time on next action',held_Train_delta_NLL_nats=-.014447927173121374,decision='Defer costly4-channel full inference until directional model is evaluated; likelihood gain alone does not establish MAE value'),
 dict(hypothesis='Memory beyond incoming heading',held_Train_delta_NLL_nats=-.021583516789796323,decision='Potential recurrent/prefix alternative if current structural model fails; must generate prefix at inference'),
 dict(hypothesis='Neural local dynamics underfit versus empirical conditioned counts',decision='Measure teacher-state transition likelihood before adding capacity; consider empirical conditional transition prior only if justified')
 ],references=[
 dict(title='Revisiting Nearest Neighbor for Tabular Data: A Deep Tabular Baseline Two Decades Later',venue='ICLR2025',url='https://proceedings.iclr.cc/paper_files/paper/2025/file/ecd294194d5155432aab533f7ce0f6fe-Paper-Conference.pdf',used_for='Learned metric, stochastic neighbors, supervised label aggregation; our residual MAE variant differs'),
 dict(title='UNITE',venue='KDD2026',url='https://web.comp.polyu.edu.hk/csmlyiu/conf/KDD26_UNITE.pdf',local_official_code='propuesta/referencias/UNITE',commit='ca60d6d86e24b92e39c70537c96a166636c90af1',used_for='Joint route/time generation and recurrent progression; not a Porto result and not a claim to match published protocol')
 ],candidate_preservation_threshold_s=131.06,Test='sealed until frozen candidate',status='ACTIVE; do not stop after a failed batch'))
print('Architecture, complete ablation conclusion and preservation policy recorded.')

