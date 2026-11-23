from base_replanteo import *
r=read(OUT/'tipo_latente_cpu_s0/resultado.json')
assert not r['qualified_candidates'] and all(x['validation']['mae_s']>131.25671207822356 for x in r['outputs'].values())
targets=[OUT/'tipo_latente_cpu_s0/matched_unsupervised.pt',OUT/'tipo_latente_cpu_s0/supervised_type.pt',OUT/'piloto_tipo_condicional_s0/cabezas.pt',OUT/'diagnostico_tipo_latente_s0/clasificador.cbm']
removed=[]
for p in targets:
 resolved=p.resolve();assert resolved.is_relative_to(OUT.resolve())
 if p.exists():
  size=p.stat().st_size;assert size<20*1024**2
  removed.append(dict(file=str(p.relative_to(ROOT)),bytes=size,sha256=sha(p),reason='Discarded or reconstructible intermediate; full semantic duration heads did not improve protected reference'))
  p.unlink()
for v in r['outputs'].values():v['weights_retained']=False;v['weights_reason']='Full Validation did not improve reference; metrics, all predictions, exact source and training protocol retained'
r['discarded_weights_audit']=removed
write(OUT/'tipo_latente_cpu_s0/resultado.json',r)
for folder,key in [('piloto_tipo_condicional_s0','heads_retained'),('diagnostico_tipo_latente_s0','classifier_retained')]:
 p=OUT/folder/'resultado.json';v=read(p);v[key]=False;v['cleanup_note']='After full-cohort descendant failed, reconstruction sources and all metrics/predictions retained; small unused intermediate weights removed.';write(p,v)
write(OUT/'limpieza_tipo_latente.json',dict(removed=removed,total_bytes=sum(v['bytes'] for v in removed),all_prediction_archives_retained=True,protected_reference_untouched=True,TransferTraj_untouched=True))
c=read(OUT/'CONTINUIDAD_OPERATIVA.json');c['CPU_trial']['weights_retained']=False;c['next_conditional_GPU_family'].update(supervisor_pid=8372,session=69122,stage='Waiting for actual risk process and supervisor exit; no second GPU',sources_frozen=True);c['next_after_result']=[
 'Read riesgo_total_s0 full result and status. Partial CRPS and512 selection are not primary MAE.',
 'Supervisor8372/session69122 waits original risk34284 and supervisor29428 handles. If risk improves, it stops queue to prioritize reproduction. Otherwise, after complete risk result it starts the matched auxiliary request-type arms sequentially.',
 'Do not edit queued aux sources or active risk sources. Check pipeline_aux_solicitud_estado.json and plan_aux_solicitud.json.',
 'CPU frozen semantic tilt completed:134.7878 vs135.1058 control, worse than131.2567 reference. No more blending of these frozen outputs. Weights discarded; metrics/predictions/source retained.',
 'Joint auxiliary label is a different training objective on the original duration components, not another frozen residual density head. Its CPU probability/gradient gate passed; actual GPU gate still mandatory.',
 'Preserve every full candidate below131.06 and any new best; require raw reproduction before claims. Test sealed and confirmation73.5 objective still pending.',
 'After each outcome, draw a concrete conclusion and continue meaningful distinct improvement rather than stopping goal after a batch.'
]
write(OUT/'CONTINUIDAD_OPERATIVA.json',c)
print(json.dumps(dict(removed_files=len(removed),removed_MiB=sum(v['bytes'] for v in removed)/1024**2,goal='active',Test='sealed')))
