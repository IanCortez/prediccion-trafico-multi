"""Register concrete active globalposterior experiment; no held metrics read before fit ends."""
from base_replanteo import *
p=OUT/'CONTINUIDAD_OPERATIVA.json';s=read(p)
proof=read(OUT/'verificacion_distribucion_cuantiles_ruta.json');assert proof['passed'] and proof['dataset_rows_timestamps_original_labels_exact']
plan=read(OUT/'piloto_posterior_ruta_global_train_s0/plan.json');codes=read(OUT/'codigos_ruta_global_train_s0/resultado.json')
s['current_trial']=dict(experiment='piloto_posterior_ruta_global_train_s0',stage='CPU2fixedstagefit, publictime-only/semanticglobalposterior/matchedrandomposterior; noheldscores yet.',pid=5152,session_id=50326,log='piloto_posterior_ruta_global_train.log',Test='sealed')
s['global_posterior_evidence']=dict(dataset='Exactsame40kfit/10kheldoriginalstreetoracle, allfitarrivalsbefore1396530690.',
 independent_numeric_proof='verificacion_distribucion_cuantiles_ruta.json',
 codebook='codigos_ruta_global_train_s0/resultado.json',PCA_explained_variance=codes['PCA_explained_variance'],
 unknown_public_fit_n=codes['unknown_fit_n'],unknown_public_held_n=codes['unknown_held_n'],
 teacher_labels='16fit-onlygeometricresidualclusters plusunknowncode; noT incodeconstruction.',
 branch_query='206publicfeatures plus90publicshorteststreetsignature, sameforEVERYbranch.',
 controls=plan['branches'],gate_semantic=plan['semantic_gate'],gate_public=plan['public_only_gate'],
 no_Validation_or_Test=True)
s['previous_goal_turn_classification']='PROGRESS: completedone15epochprimarytrial andseveralTrain-onlygates, verifiednegativeoutcomes, keptallpredictions, cleanedexactintermediates; nextactionchanged.'
s['latest_storage_snapshot']=read(OUT/'inventario_conservacion.json')
s['resource_policy']='OneheavyGPUmaximum; currentjobCPU2only; MultiQuantileCPU-only, installedleafmethodExactverified. No newpackageinstallation/dense53k-squarematrix/CDFcache. Resumecompletedstageweights bySHA.'
s['next_after_result']=['FollowPID5152/session50326, immutablefrozenmainrunner/distributionhelper. Observationtimeoutsare notrestarttriggers.',
 'Completeallfixedstageforecasts, scoreheldonce. Ifsemanticpasses>=2sreference+>=1srandom+notworsepublic, certifyandexpandit.',
 'Independentlypreregisteredpublicsignature-onlybranch mayadvanceif>=2secondswithupperCI<0; do notcreditrouteauxifonlypublicbranchimproves.',
 'Ifallfail, keepmetrics/predictions andconcludeglobalposteriorconfiguration; no latentcodecountsearch. Continueobjectivefromevidence, noTestopening.',
 'Best130.158/40qualifiedmodels, originals andTransferTrajprotected;73.5confirmation unmet, goalactive.']
write(p,s);print(json.dumps(dict(recorded=True,current_trial=s['current_trial'],Test='sealed')),flush=True)
