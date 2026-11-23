"""Close failedglobalposteriorfamily, keep everyresult and register fullsequence nextstep."""
from base_replanteo import *
access=deny_test();run=OUT/'piloto_posterior_ruta_global_gpu_train_s0';r=read(run/'resultado.json');assert not r['passed'] and r['selected'] is None
z=dict(np.load(run/'predicciones_held_Train.npz'));assert sha(run/'predicciones_held_Train.npz')==r['predictions_sha256']
for mode in ['public_only','semantic','random']:assert metric(z['true'],z[mode])==r['metrics'][mode]
assert metric(z['true'],z['privileged_true_code'])==r['privileged_true_code_oracle']
retired=read(run/'pesos_descartados.json');assert len(retired['removed'])==29 and all(not (run/e['file']).exists() for e in retired['removed'])
conclusion=dict(complete=True,scope=r['scope'],metrics=r['metrics'],reference=r['reference'],comparisons=r['comparisons'],
 held_code_accuracy=r['held_code_accuracy_diagnostic'],privileged_true_code_oracle=r['privileged_true_code_oracle'],
 decision='Globalroutecodegatefailed: semantic132.615s vsrandom132.399s andpublic132.785s, belowrequired2secondgainover133.355reference; no fullprimaryexpansion/no codecountPCA/binsearch.',
 information='Fullactualroute90featureoracle94.024s; truecompressedcodeoracle118.910s. Thisfinitecoding losespredictiveinformation. Notanirreducibleerrorbound orclaimallrouteposteriorsfail.',
 resource_event='CPUsharedMultiQuantilestudyretiredforcost; completedpublic132.711s predictionrecovered. ScalarGPUstudycomplete andallthreebranchespreserved.',
 next='Fullsequenceprefixmemory/clock-awareroadmodel, no coarsecodebook. Physicalprivatepositionsrecovered; unsupportedmacroturns masked instead ofassumingentry0.',
 all_completed_metrics_predictions_sources_retained=True,unselected_weights_removed=True,Test='sealed',goal='ACTIVE')
write(OUT/'conclusion_posterior_global_gpu.json',conclusion)
idx=read(OUT/'indices_progreso_vial_s0/resultado.json');assert idx['complete']
with np.load(OUT/'indices_progreso_vial_s0/indices_Train.npz') as a:assert len(a['eligible_primary_train_index'])==1023485
write(OUT/'indices_progreso_vial_s0/ejecucion_verificada.json',dict(scientific_output_complete=True,no_rerun_needed=True,
 note='Onlyfinalconsoleprintcouldnotserializequantilearray after result/index hadbeensaved; no preprocessingfailed.',
 output_sha256=idx['outputs_sha256'],all_primary_queries_retained=True))
proof=read(OUT/'verificacion_adaptador_progreso_vial.json');assert proof['passed']
s=read(OUT/'CONTINUIDAD_OPERATIVA.json');s['current_trial']=dict(experiment='progreso_vial_recurrente_next',stage='Globaltree+neuralcodefamiliescomplete/negative. FullTrainindices andphysicalteacheradapterready/verified. Nextimplement recurrentprogress, realCPU/GPUpreflights, thenfullprimaryexperiment.',pid=None,Test='sealed')
s['completed_new_instruments']['global_route_GPU']=conclusion
s['recurrent_data_ready']=dict(index='indices_progreso_vial_s0/resultado.json',adapter='adaptador_progreso_vial_s0/resultado.json',
 verification='verificacion_adaptador_progreso_vial.json',architecture='plan_progreso_vial_recurrente.json',
 auxiliary_trip_n=894105,tokens=41775036,attachment_teacher_coverage=idx['attachment_teacher_coverage'],
 known_position_time_fraction=proof['position_known_time_fraction'],macro_labels_repaired=689590,
 corrected_legacy_noninitial_entry_errors=proof['corrected_legacy_noninitial_zero_entry_error_n'],
 no_primary_rows_or_times_changed=True,do_not_use_legacy_masks_as_macro_truth=True,
 public_D=dict(shape=[23039,23039],bytes=2123182084,persistent_in_checkpoints=False))
s['next_after_result']=['Implementactualrecurrentprefix/clock-awaremodel onfullTrainroads, notanothercodecount/featurebinvariant. Architecture draftandvalidatedDataindex/adapter exist.',
 'Useall1023485originalpurgedTrainqueries forprimaryT;894105roadteacherparents partial/maskedaux. No queryactualroad/N/GTtime.',
 'Useadapterpositions/macro/reset/EOS flags, notlegacynext_route_valid/assumedentry0. 689590legacymacrofalsepositives; time labels untouched.',
 'Keep2.12GBpublicD NONPERSISTENT, no modelobjectcheckpoint; assertstatepayload<50MiB. OneGPU.5onlyaftermeaningfulstresspreflight.',
 'RawqueryadapterkeepsoriginalpostTrain-dateguard; rawpublic296features testedexactlywithoutValidationlabels. CUA not involved.',
 'Everycompletedexperimentmetric/predictionpreservedincluding>131.06;40qualifiedmodels/originals/best/TransferTrajprotected.73.5unmet, Testsealed.']
write(OUT/'CONTINUIDAD_OPERATIVA.json',s)
# Latergeometryaudit limits olderauxiliaryinterpretation, while primarypointmetric staysvalid.
caveat=read(OUT/'conclusion_valor_vial.json');caveat['later_auxiliary_geometry_caveat']=dict(
 file='verificacion_adaptador_progreso_vial.json',finding='Legacychainendpointadjacency didnotguarantee entry0 afterinteriorrawnode reversal.',
 primary_metric_still_valid=True,claim_that_this_caused_failure=False,new_adapter_required_forfutureprogress=True)
write(OUT/'conclusion_valor_vial.json',caveat)
print(json.dumps(dict(concluded=True,semantic=r['metrics']['semantic'],random_control=r['metrics']['random'],full_sequence_data_verified=True,goal='ACTIVE')),flush=True)
