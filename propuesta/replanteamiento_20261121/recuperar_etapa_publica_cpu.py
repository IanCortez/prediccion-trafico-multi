"""Retire slowCPUstudy afterverifiedresourcecost; recover everycompletedpublicstage prediction."""
from base_replanteo import *
from catboost import CatBoostRegressor
from inferencia import feature_input
from distribucion_cuantiles_ruta import rearrange,mixture_quantiles
from numba import set_num_threads
access=deny_test();start=time.time();set_num_threads(2);run=OUT/'piloto_posterior_ruta_global_train_s0'
assert not (run/'resultado.json').exists();manifest=read(run/'etapas_completas.json');assert set(manifest)=={'public_time_only'}
path=run/manifest['public_time_only']['file'];assert sha(path)==manifest['public_time_only']['sha256']
model=CatBoostRegressor();model.load_model(str(path));assert model.tree_count_==1000
source=OUT/'oraculo_tiempo_calle_train_s0'
with np.load(source/'auxiliares_Train.npz') as z:q={key:z[key] for key in ['row','train_index','timestamp','true']}
codefolder=OUT/'codigos_ruta_global_train_s0';codes=dict(np.load(codefolder/'codigos_y_firmas_Train.npz'))
for key in ['row','timestamp','train_index']:assert np.array_equal(q[key],codes[key])
d,X,names=feature_input('train',read(BEST/'resultado.json'));keep=np.asarray([not n.startswith('mm_') for n in names])
X=np.column_stack([X[q['train_index']][:,keep],codes['public_signature'].astype(np.float32)]);held=X[40000:]
cq=rearrange(model.predict(held,thread_count=2));f=mixture_quantiles(cq[:,None],np.ones((len(held),1)))
assert np.max(np.abs(f[:,1]-cq[:,4]))<1e-6
r=dict(np.load(source/'predicciones_held_Train.npz'));y=q['true'][40000:]
for key in ['row','timestamp']:assert np.array_equal(r[key],q[key][40000:])
assert np.array_equal(r['true'],y)
prediction_file=run/'predicciones_etapa_publica_held_Train.npz';np.savez_compressed(prediction_file,row=q['row'][40000:],timestamp=q['timestamp'][40000:],true=y,pred=f[:,1],quantiles=f,reference=r['control'])
comparison=paired(y,f[:,1],r['control'],q['timestamp'][40000:]);comparison['interpretation']='Completedpublic-onlystage recoveredfromcomputationallyretiredglobalstudy; no semantic/randomcomparisonfinished.'
passed=bool(comparison['delta_mae_s']<=-2 and comparison['ci95_s'][1]<0)
result=dict(complete_public_stage=True,complete_global_study=False,public_branch_gate_passed=passed,metrics=metric(y,f[:,1]),reference=metric(y,r['control']),comparison=comparison,
 model_sha256=manifest['public_time_only']['sha256'],model_file=str(path),predictions_sha256=sha(prediction_file),source_sha256=sha(__file__),
 no_Validation_or_Test_access=True,seconds=time.time()-start)
write(run/'resultado_etapa_publica.json',result)
retired=dict(retired=True,do_not_restart_this_CPU_runner=True,
 reason='Resourceimplementationreplacementbeforeobservinganycompletedheldmetric: CPUfirstsharedMultiQuantile1000trees1072seconds, verifiedliveafter1322seconds/2605CPUseconds. GPUscalar1000trees~29seconds perquantile, all9~270seconds, samequeries/controls preregisteredbeforeCPUscore.',
 first_completed_stage_wall_s=1072.3221309185028,verified_live_snapshot=dict(elapsed_s=1322.6737346,cpu_s=2605.171875,pid=5152),
 process_identity_verified_before_stop=True,original_source_and_protocol_unchanged=True,
 completed_models=['public_time_only'],unfinished_stage='semantic_posteriorclassifier, stoppedbeforefinalmodelpersisted',
 every_completed_stage_prediction_recovered=True,globalcomparison_not_claimed=True,
 replacement='piloto_posterior_ruta_global_gpu_train_s0',replacement_difference='Nineindependentquantiletreesets instead ofsharedmultioutputtrees. Matchedcapacitywithinreplacement.',
 stopped_for_resource_efficiency_not_MAE=True,Test='sealed')
write(run/'retiro_por_recursos.json',retired)
if not passed:
 assert sha(path)==manifest['public_time_only']['sha256'];size=path.stat().st_size;path.unlink()
 write(run/'pesos_descartados.json',dict(removed=[dict(file=path.name,bytes=size,sha256=manifest['public_time_only']['sha256'])],all_completed_predictions_preserved=True))
print(json.dumps(dict(completed_public_stage=True,complete_global_study=False,metrics=result['metrics'],reference=result['reference'],public_gate_passed=passed)),flush=True)
