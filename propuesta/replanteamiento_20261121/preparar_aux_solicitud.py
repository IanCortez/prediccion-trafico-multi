from base_replanteo import *
p=OUT/'entrenar_aux_solicitud.py';s=p.read_text(encoding='utf-8-sig');s=s.replace("torch.cuda.set_rng_state(resume['cuda_rng'])","torch.cuda.set_rng_state(resume['cuda_rng'].cpu())");p.write_text(s,encoding='utf-8')
compile(s,str(p),'exec')
r=read(OUT/'tipo_latente_cpu_s0/resultado.json')
assert r['Validation_n']==127941 and not r['qualified_candidates'] and r['reference_CPU_equal_protected']
conclusion=dict(reference_MAE_s=131.25671207822356,unsupervised_MAE_s=r['outputs']['matched_unsupervised']['validation']['mae_s'],semantic_MAE_s=r['outputs']['supervised_type']['validation']['mae_s'],semantic_vs_control=r['paired_semantic_vs_matched'],versus_reference=r['outputs']['supervised_type']['paired_vs_reference'],qualified=[],interpretation='Semantic label provides a small matched-control gain, but both frozen residual density heads degrade the strong reference. Train-reserved-head pilot was optimistic for absolute performance; its frozen representation already saw all Train. Do not blend/search weights of these frozen outputs.',next_hypothesis='Test label as conditional auxiliary likelihood on original route-duration components, preserving the original duration marginal and original primary/geometry objectives. Avoid fitting a correction density to in-sample reference errors. This remains a hypothesis, not a promised gain.',Test='sealed')
write(OUT/'conclusion_tipo_latente_cpu.json',conclusion)
plan=dict(hypothesis=conclusion['next_hypothesis'],factorization='p(T,c|x)=sum_z pi_z(x) p_z(T|x) q_z(c|x). Sum_c p(T,c|x)=p(T|x).',
 training='Original NLL +5RPS +0.2 conditional geometry NLL +lambda*(-log p(c|T,x)); lambda 0 matched control or0.2 semantic arm.',
 query='Original OD/time duration marginal. Actual CALL_TYPE and duration enter only training losses, never forward.',
 initialization='Exact protected duration model; zero class-head logits. Duration predictions unchanged at step0.',
 gate='CPU arithmetic/gradient gate passed. Mandatory actual GPU finite-gradient gate and exact whole-Validation reference before updates.',
 budget='4 fixed epochs each arm, primary LR1e-5, new class head LR1e-3, 512 batch, FP32 highest, no blending-weight search.',
 selection='All four full Validation outputs retained. Preserve every qualifying checkpoint <131.06s and every new best. Epoch0 means original reference and is never called a learned gain.',
 ordering='Wait for actual risk trainer AND supervisor exits. Require full risk result; reproduce any improved risk candidate first. Then execute one GPU arm at a time.',
 limitations='Repeated adaptive Validation is exploratory. Train call-type gain is small. Joint auxiliary may still overfit and fail. No cohort changes or Test access.',
 sources={f:sha(OUT/f) for f in ['entrenar_aux_solicitud.py','modelo_aux_solicitud.py','verificar_aux_solicitud_cpu.py']})
write(OUT/'plan_aux_solicitud.json',plan)
c=read(OUT/'CONTINUIDAD_OPERATIVA.json');c['CPU_trial']=dict(experiment='tipo_latente_cpu_s0',complete=True,qualified_candidates=[],conclusion=conclusion);c['next_conditional_GPU_family']=dict(plan='plan_aux_solicitud.json',waiting_on='riesgo_total_s0 and supervisor',GPU_jobs=1,CPU_gate_passed=True)
write(OUT/'CONTINUIDAD_OPERATIVA.json',c)
marker='<!-- cierre_tipo_latente_y_aux_conjunta -->'
text="""

<!-- cierre_tipo_latente_y_aux_conjunta -->
La evaluación completa de cabezas congeladas terminó. Control sin etiqueta: MAE 135,105752 s; tipo supervisado: 134,787793 s. La etiqueta mejora su control en 0,317959 s [IC95 % exploratorio por día: -0,419840; -0,219175], pero empeora la referencia en 3,531081 s [2,517468; 4,630605]. Ningún candidato nuevo. La referencia CPU reprodujo exactamente todas las predicciones protegidas; también pasaron orden inverso y consultas individuales. No se ajustarán más mezclas para estas dos salidas.

La señal pequeña a favor de la etiqueta motiva un ensayo distinto: aprender p(T,c|OD) con el tipo como pérdida auxiliar condicional sobre los ocho componentes originales, manteniendo su marginal de duración, su objetivo primario y su supervisión geométrica. Al consultar se suma sobre el tipo; la llamada real no es entrada. Este ensayo no depende de una corrección ajustada a errores internos de la referencia. La prueba CPU de factorizar/marginalizar y transmitir gradientes pasó, sin actualizar pesos. Es una hipótesis aún sin MAE.

Se prepararon cuatro épocas fijas con/sin esa pérdida, con prueba GPU inicial obligatoria y una única GPU. Se espera el resultado y la salida real del ensayo de riesgo; cualquier mejora suya tendrá prioridad para reproducción. Ver plan_aux_solicitud.json y pipeline_aux_solicitud_estado.json. El objetivo <73,5 s permanece activo y Test cerrado.
"""
for f in [OUT/'README.md',BASE/'RESUMEN_RESULTADOS.md']:
 body=f.read_text(encoding='utf-8-sig')
 if marker not in body:f.write_text(body+text,encoding='utf-8')
print('Resultado completo registrado y control auxiliar preparado, sin otra GPU activa')
