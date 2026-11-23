from base_replanteo import *
native=OUT/'prefijo_nativo_s0'
r=read(native/'resultado.json')
assert r['validation']['n']==127941 and r['selected_alpha']==0 and not r['qualified_candidates']
p=np.load(native/'validacion.npz')
old=np.load(BEST/'validacion.npz')
assert np.array_equal(p['row'],old['row']) and np.array_equal(p['pred'][0],old['pred'])
comparisons=[]
for i,a in enumerate(r['config']['alphas']):
    comparisons.append(dict(alpha=a,validation=r['all_alpha_validation'][i],paired_vs_reference=paired(p['true'],p['pred'][i],old['pred'],p['timestamp'])))
pilot=read(OUT/'piloto_historia_condicional_s0/resultado.json')
assert not pilot['worth_GPU_test']
write(OUT/'conclusion_nativo_completa.json',dict(n=127941,outputs=comparisons,selected_alpha=0,selected_output_is_exact_original_reference=True,no_new_primary_best=True,preservation_threshold_reached=False,interpretation='Autoregressive teacher-forced likelihood improved in Train, but none of these complete outputs improved primary MAE. Do not extend this same training or rerun blend searches. Test full-trip marginal CRPS next.',historical_prior_rejected=dict(NLL_history=pilot['results']['history']['held_NLL'],NLL_native=pilot['results']['native']['held_NLL'],NLL_matched_control=pilot['results']['control']['held_NLL'],difference=pilot['history_minus_matched_control'],scope='Only new heads held out; frozen representation saw Train. Reject this configuration, not every use of temporal history.'),Test='sealed',goal='active'))
risk=OUT/'riesgo_total_s0'
v=read(risk/'verificacion_referencia.json')
assert v['historical_parent'] is False and v['parent_sha256']==r['model_sha256']
assert read(risk/'verificacion_GPU_inicial.json')['passed']
# Remove only completed/unused intermediates; current parent weights stay intact.
removed=[]
for target in [native/'ultima_reanudacion.pt',OUT/'prior_eventos/probabilidad_train.npz']:
    assert target.resolve().is_relative_to(OUT.resolve())
    if target.exists():
        assert target.stat().st_size<50*1024**2
        removed.append(dict(path=str(target.relative_to(ROOT)),bytes=target.stat().st_size,sha256=sha(target)))
        target.unlink()
write(OUT/'limpieza_nativo_y_control_historia.json',dict(removed=removed,reason='Native training+full evaluation complete; best native checkpoint needed by active risk trial remains. Temporal GPU gate rejected; its full probability lookup is an unused intermediate. Compact count table and reproducible sources remain.',current_risk_weights_untouched=True,best_reference_untouched=True,TransferTraj_untouched=True,original_data_untouched=True))
meta=read(OUT/'prior_eventos/resultado.json')
meta['outputs']['probabilidad_train.npz']['retained']=False
meta['outputs']['probabilidad_train.npz']['reason']='Rejected temporal GPU configuration; reconstructible from retained conteos_fold.npz with historia_eventos.py'
write(OUT/'prior_eventos/resultado.json',meta)
call=OUT/'diagnostico_tipo_latente_s0'
c=read(call/'resultado.json')
c['input_hashes']={p.name:sha(p) for p in [DATA/'train_x.npz',OUT/'datos_semi_markov/runs_train.npz']}
c['predictions_sha256']=sha(call/'held_Train_predicciones.npz')
if (call/'clasificador.cbm').exists():c['classifier_sha256']=sha(call/'clasificador.cbm')
c['next_scope']='Auxiliary metadata is predictable from allowed OD/time variables. A future supervised latent-type duration model must infer/marginalize type at query; never set enriched=True or pass observed call. No travel-time experiment or MAE gain yet.'
write(call/'resultado.json',c)
state=read(OUT/'CONTINUIDAD_OPERATIVA.json')
state['active_unified_session']=14113
state['current_trial']=dict(experiment='riesgo_total_s0',pid=34284,supervisor_pid=29428,session=14113,device='GPU',GPU_gradient_gate_passed=True,parent='prefijo_nativo_s0',parent_sha256=r['model_sha256'],do_not_modify_running_sources=True)
state['latest_complete_native']=read(OUT/'conclusion_nativo_completa.json')
state['temporal_history_GPU_trial_rejected']=True
state['cleanup_after_native']=removed
state['call_auxiliary_diagnostic']=c
state['do_not_restart_finished_processes']=[35928,34908]
state['goal_status']='ACTIVE; no primary improvement yet; Test sealed'
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
print(json.dumps(dict(native_MAE=r['validation']['mae_s'],largest_generator_weight_MAE=r['graph_validation']['mae_s'],temporal_GPU_skipped=True,risk_GPU_initial_gate_passed=True,removed_MiB=sum(z['bytes'] for z in removed)/1024**2,call_classifier_accuracy=c['held_Train_classification']['accuracy'],goal='active')))
