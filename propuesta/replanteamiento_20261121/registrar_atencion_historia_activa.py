"""Protect live branch checkpoints and align continuity with the actual next experiment."""
from base_replanteo import *
run=OUT/'atencion_historia_individual_s0';proof=read(run/'verificacion_GPU_inicial.json');assert proof['passed']
assert read(run/'verificacion_referencia.json')['full127941point_predictions_exact']
live=read(run/'candidatos_conservados.json')
for item in live['candidates']:
 path=(run/item['file']).resolve();assert path.is_relative_to(run.resolve()) and sha(path)==item['sha256']
state=read(OUT/'CONTINUIDAD_OPERATIVA.json')
state['current_trial']=dict(experiment=run.name,pid=27364,exec_session_id=71054,
 stage='GPUtraining8fixed frozen-parent epochs;186848branch parameters; source64tokens pooled into8newcomponents; Testsealed.',
 sources_frozen=True,query='same five fields; all source labels from eligibleTrain, ownqueryfold excluded.',
 parent_reference='historica_fresca_s0 epoch27 certified130.204821s',
 initial_new_candidate_mae_s=130.21818650784346,initial_candidate_checkpoint_preserved=True,
 note='Initialcandidate is not a learned gain; global best unchanged until completed and certified.')
state['full_prediction_audit']=dict(outputs=213,archives=26,maximum_reported_metric_error=0,all_rows_labels_timestamps_exact=True)
state['individual_history_input_bank']=dict(MiB_including_shared_memory=302.1787462234497,cap_MiB=450,
 Train_only_normalization_queries=32768,source_tokens_used_for_normalization=1996111,raw_cached_Train_tokens_exact=True,
 ownfold_time_geometry_intervention_exact=True,source_order_logp_error=0.,missing_history_parent_fallback_exact=True)
state['next_after_result']=['Observe actual attentionPID27364/session71054 and atencion_historia_individual_s0.log; do not restart solely on a timeout.',
 'Do not edit active trainer/model/token helper/base/inference-history/aux sources.8epochs,1800s cap, oneGPU6GiBcap; checkpoint branch only.',
 'At terminal audit all full127941initial/epoch/selected point predictions. Preserve every qualifying branch checkpoint and original31historical weights, originalbest,TransferTraj.',
 'If fullValidation improves parent by at least.1s and exploratorydayCIupper<0: implement frozen individual-attention inference, certify cachedfull/raw65/order/singleton using SAME GPUtoken constructor, then run --mode compressed with unchanged source/init/batches.',
 'Compression control replaces sourceG/path/time with nearest6prototype summaries, keeps64source relation fields, matched architecture. Inspect numeric initialization hash and fixedepochs before causal attribution.',
 'If negative/inconclusive, recordcritically and choose a genuinely different next experiment; do not repeat fine-clock/directmedian families. Target73.5confirmation remains unmet; Testclosed.']
state['previous_goal_turn_classification']='PROGRESS: full historical control certified,31modelsprotected;6epochdirectMAEnegativecompleted/cleaned;64historycompressiongatepositive; indices302MiBbounded,Trainnormfit,raw/exclusion/permutation/fallback/gradproofs passed, actual frozen-parentattentionGPUtraining launched.'
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
p=OUT/'verificar_conservacion.py';s=p.read_text(encoding='utf-8-sig')
if 'individual_history_branch_checkpoints' not in s:
 needle="write(OUT/'inventario_conservacion.json',report);"
 block="""report['individual_history_branch_checkpoints']=[]
for name in ['atencion_historia_individual_s0','control_compresion_historia_s0']:
 manifest=OUT/name/'candidatos_conservados.json'
 if manifest.exists():
  for candidate in read(manifest)['candidates']:
   file=(manifest.parent/candidate['file']).resolve()
   assert file.is_relative_to(manifest.parent.resolve()) and sha(file)==candidate['sha256']
   report['individual_history_branch_checkpoints'].append(dict(id=name,epoch=candidate['epoch'],mae_s=candidate['validation']['mae_s'],sha256=candidate['sha256'],unchanged=True))
write(OUT/'inventario_conservacion.json',report);"""
 assert s.count(needle)==1;s=s.replace(needle,block);p.write_text(s,encoding='utf-8')
for p in [OUT/'README.md',BASE/'RESUMEN_RESULTADOS.md']:
 t=p.read_text(encoding='utf-8-sig')
 old='La siguiente vía examina los 64 viajes históricos individuales: su diagnóstico de cobertura geométrica en Train pasó (24,7 %), sin demostrar todavía mejora del MAE.'
 new='La atención sobre 64 viajes históricos individuales ya está entrenándose con el modelo previo congelado y 186.848 parámetros nuevos. Su diagnóstico geométrico de Train pasó (24,7 %); la entrada, exclusión de fuentes, invariancia y gradientes se verificaron antes del entrenamiento. La evaluación inicial fue 130,218187 s, todavía sin ganancia aprendida. El banco ocupa 302,18 MiB contando la memoria compartida; sus checkpoints guardan solo la rama nueva.'
 assert old in t;t=t.replace(old,new);p.write_text(t,encoding='utf-8')
print(json.dumps(dict(active_trial=run.name,source_frozen=True,qualified_branch_models_protected=len(live['candidates']),goal='ACTIVE')))
