from base_replanteo import *
state=read(OUT/'CONTINUIDAD_OPERATIVA.json')
run=OUT/'subtrayectos_conjuntos_s0'
for name in ['preflight_32.json','preflight_512.json','preflight_consulta_2048.json']:
 assert read(run/name)['passed']
state['current_trial']=dict(experiment=run.name,stage='Trainingfixed24epochs, fulljointfreshweights; realTrain gradients and canonical2048inference memory preflights passed.',
 pid=37148,exec_session=4915,log='entrenar_subtrayectos_conjuntos.log',Test='sealed',
 current_history=str((run/'historia.json').relative_to(OUT)),sources_frozen=True)
state['latest_storage_snapshot']=read(OUT/'inventario_conservacion.json')
state['resource_policy']='OneheavyGPU .5cap, TrainB512 measured5772MiB, canonicalinference2048 measured6130MiB beforetraining; allTrain tensors releasedbeforefinalevaluation.'
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
policy=read(OUT/'politica_recursos.json')
policy['active_joint_fresh']=dict(Train_batch=512,inference_batch=2048,GPU_memory_fraction=.5,
 measured_Train_reserved_MiB=5772,measured_inference_reserved_MiB=6130,
 explicit_Train_tensor_release_before_evaluation=True,one_heavy_GPU=True,no_new_environment_installation=True)
write(OUT/'politica_recursos.json',policy)
print('Activeprocess and verifiedresourcefigures registered.')
