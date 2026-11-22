from base_replanteo import *
import ast
p=OUT/'entrenar_riesgo_total.py'
assert not (OUT/'riesgo_total_s0').exists()
s=p.read_text(encoding='utf-8-sig')
changes={
"conditional_crps_surrogate(smoke['times'][ok],target[ii][ok],smoke['log_probability'][ok])":"conditional_crps_surrogate(smoke['times'][ok].double(),target[ii][ok].double(),smoke['log_probability'][ok])",
"conditional_crps_surrogate(sample['times'][ok],target[ii][ok],sample['log_probability'][ok])":"conditional_crps_surrogate(sample['times'][ok].double(),target[ii][ok].double(),sample['log_probability'][ok])",
"history=[];best=metric(v[proxy],initial[-1])['mae_s'];best_step=0;bad=0;first_step=0;previous_seconds=0.":"history=[];best=metric(v[proxy],initial[-1])['mae_s'];best_step=0;bad=0;first_step=0;previous_seconds=0.;failed_total=0;query_total=0;finished=False",
"rng.bit_generator.state=saved['numpy_generator'];torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])":"rng.bit_generator.state=saved['numpy_generator'];torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])\n        failed_total=saved['failed_total'];query_total=saved['query_total'];finished=saved['finished']",
"def save_resume(step):":"def save_resume(step,finished=False):",
"elapsed=previous_seconds+time.time()-start,numpy_generator=rng.bit_generator.state":"elapsed=previous_seconds+time.time()-start,failed_total=failed_total,query_total=query_total,finished=finished,numpy_generator=rng.bit_generator.state",
"last_step=first_step;failed_total=0;query_total=0\n    for step in range(first_step+1,args.steps+1):":"last_step=first_step\n    training_steps=[] if finished else range(first_step+1,args.steps+1)\n    for step in training_steps:",
"surrogate+=float(loss.detach())/4;risk+=float(score)*600/4":"surrogate+=float(loss.detach())/4;risk+=float(score)*600*n",
"sampled_CRPS_s=risk,score_surrogate":"sampled_CRPS_mean_over_complete_queries_s=risk/max(16-failed_step,1),score_surrogate",
"    save_resume(last_step)\n    ck=torch.load":"    save_resume(last_step,finished=True)\n    ck=torch.load"
}
for a,b in changes.items():
    assert a in s,a
    s=s.replace(a,b)
ast.parse(s);p.write_text(s,encoding='utf-8')
r=read(OUT/'plan_riesgo_total_generado.json')
r['sampler_implemented']=dict(source='muestreador_rechazo_nativo.py',sha256=sha(OUT/'muestreador_rechazo_nativo.py'),synthetic_verification=read(OUT/'verificacion_muestreador_rechazo.json'),actual_Train_CPU_probe=read(OUT/'sondeo_riesgo_total_cpu.json'))
r['trainer_prepared']=dict(source=p.name,sha256=sha(p),not_started=True,query_microbatch=4,accumulation=4,accepted_samples=8,max_steps=4000,lr=.00001,teacher_weight=.1,canonical_inference=128,requires_actual_GPU_gradient_gate_before_long_run=True,policy_sampling='Encoder dropout disabled, GRU train mode/dropout0, count priors exclude current Train fold',finite_budget_fallback='Failed conditional samples use teacher loss on the same query; record rate, no primary cohort exclusion',resume='Finished checkpoint flag prevents retraining when only inference needs recovery')
r['status']='Estimator and sampler verified mathematically, synthetic sampler and four real Train CPU gradients passed; full-trial source prepared, not started, actualGPU gate pending'
write(OUT/'plan_riesgo_total_generado.json',r)
print('Riesgo total preparado con reanudación, registro de soporte y prueba GPU obligatoria.')
