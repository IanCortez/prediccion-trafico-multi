from base_replanteo import *
import ast
p=OUT/'entrenar_riesgo_total.py'
assert not (OUT/'riesgo_total_s0').exists()
s=p.read_text(encoding='utf-8-sig')
a="ii,smoke=draw(eligible[:4],particles=4,candidates=8)"
b="smoke_sequences=np.random.default_rng(2026).choice(eligible,4,replace=False)\n    ii,smoke=draw(smoke_sequences,particles=4,candidates=8)"
assert a in s;s=s.replace(a,b)
a="sampled_CRPS_mean_over_complete_queries_s=risk/max(16-failed_step,1),score_surrogate"
b="sampled_CRPS_mean_over_complete_queries_s=(risk/(16-failed_step) if failed_step<16 else None),complete_policy_queries=16-failed_step,score_surrogate"
assert a in s;s=s.replace(a,b)
a="finite_nonzero_complete_path_gradients=True,encoder_dropout_disabled=True"
b="finite_nonzero_complete_path_gradients=True,smoke_Train_rows=sequences.trip[smoke_sequences].tolist(),encoder_dropout_disabled=True"
assert a in s;s=s.replace(a,b)
ast.parse(s);p.write_text(s,encoding='utf-8')
r=read(OUT/'plan_riesgo_total_generado.json')
r['trainer_prepared']['sha256']=sha(p)
r['implementation_review']=dict(no_completed_policy_samples_metric_is_null_not_zero=True,GPU_smoke_same_Train_query_selection_as_passed_CPU_probe=True,finished_checkpoint_prevents_extra_training_on_inference_resume=True,teacher_fallback_does_not_remove_primary_queries=True,source_not_running_when_edited=True,actual_GPU_validation_pending=True)
write(OUT/'plan_riesgo_total_generado.json',r)
write(OUT/'revision_guardias_riesgo.json',dict(passed_static_review=True,changes=r['implementation_review'],trainer_sha256=sha(p),Test='sealed',goal='active'))
print('Guardias revisadas: ausencia de muestras se registra como dato ausente, y prueba GPU alineada con Train comprobado.')
