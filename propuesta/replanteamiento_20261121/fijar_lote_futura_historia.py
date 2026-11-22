from base_replanteo import *
import ast
p=OUT/'entrenar_prefijo_historia.py'
assert not (OUT/'prefijo_historia_s0').exists()
s=p.read_text(encoding='utf-8-sig')
changes={
'canonical_query_batch=16':'canonical_query_batch=128',
'for first in range(0,len(indices),16):':'for first in range(0,len(indices),128):',
'indices[first:first+16]':'indices[first:first+128]',
'if n<16:ii=torch.cat([ii,ii[-1:].expand(16-n)])':'if n<128:ii=torch.cat([ii,ii[-1:].expand(128-n)])',
"    model.train()\n    initial=batch_loss(eligible[:8])":"    warmup,_=predict(proxy[:128])\n    assert np.isfinite(warmup).all()\n    assert np.array_equal(warmup[0],reference_pred[proxy[:128]])\n    model.train()\n    initial=batch_loss(eligible[:8])",
'finite_actual_Train_gradients=True,reference_reproduced=True':'finite_actual_Train_gradients=True,reference_reproduced=True,canonical128_inference_warmup_passed=True'
}
for a,b in changes.items():
    assert a in s,a
    s=s.replace(a,b)
s=s.replace("comparison_limit='Warm-start","inference_batch_scope='Canonical128 fixed before training; no cross-batch equality claim for stochastic paths. Validate query order and singleton at this canonical shape before promotion.',comparison_limit='Warm-start")
ast.parse(s);p.write_text(s,encoding='utf-8')
r=read(OUT/'preparacion_entrenador_historia.json')
r.update(target_sha256=sha(p),canonical_query_batch=128,scope='A new numerical configuration declared before this trial. Current native model retains16; larger batches change some sampled paths through finite precision, although seeds and particle count are unchanged. Reference stays canonical2048.')
write(OUT/'preparacion_entrenador_historia.json',r)
# Clarify the accepted-category index in the mathematical test only.
p=OUT/'verificar_gradiente_crps_condicional.py'
s=p.read_text(encoding='utf-8-sig').replace('The third unconditional category is outside the cohort.','Unconditional category0 is outside the cohort; categories1and2 are eligible.')
p.write_text(s,encoding='utf-8')
c=read(OUT/'CONTINUIDAD_OPERATIVA.json')
c['replacement_inference_batch_result']=read(OUT/'prefijo_nativo_s0/evaluacion_acelerada/seleccion_lote.json')
c['replacement_inference_outcome']='Larger batches are faster but change stochastic trajectories; keep original canonical16. No acceleration claimed. Replacement now saves point-prediction progress every8192rows and verifies canonical query order/singletons.'
c['history_future_canonical_batch']=128
c['history_future_inference_scope']=r['scope']
c['current_history_pipeline']=dict(pid=34908,session=12734,script='continuar_tras_prefijo_acelerado.py',waiting_for_pid=35928)
c['superseded_sessions']=[9112,46174]
c['next_pipeline']='Follow current34908 supervisor/session12734, not old35436. CPU pilot remains conditional on complete frozen parent result. HistoricalGPU only if pilot beats both controls.'
write(OUT/'CONTINUIDAD_OPERATIVA.json',c)
print('Native mantiene16; futura variante fija128 con preflight propio y límites declarados.')
