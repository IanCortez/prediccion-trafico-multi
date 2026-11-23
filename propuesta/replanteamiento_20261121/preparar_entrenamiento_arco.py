"""Separate semanticcorrection experiment, source-frozen before newValidation outcome."""
from base_replanteo import *
import ast
source=OUT/'atencion_subtrayectos_s0/fuentes/entrenar_atencion_subtrayectos.py'
assert sha(source)=='64a4af8b15db701cb749396778597ea633c7a0fc9c075861bcb6b9e79f29bf1a'
s=source.read_text()
s=s.replace('from tokens_subtrayectos import SubtripTokenBank,query_parts_sub',
 'from tokens_subtrayectos import query_parts_sub\nfrom tokens_subtrayectos_arco import SubtripArcLengthBank')
s=s.replace("normal=read(folder/'normalizacion_resultado.json')","normal=read(OUT/'constructor_tokens_arco_s0/resultado.json')")
s=s.replace("sha(folder/'constructor_tokens.npz')","sha(OUT/'constructor_tokens_arco_s0/constructor_tokens.npz')")
s=s.replace("constructor=dict(np.load(folder/'constructor_tokens.npz'))","constructor=dict(np.load(OUT/'constructor_tokens_arco_s0/constructor_tokens.npz'))")
s=s.replace("run=OUT/'atencion_subtrayectos_s0'","run=OUT/'atencion_subtrayectos_arco_s0'")
s=s.replace("'entrenar_atencion_subtrayectos.py'","'entrenar_atencion_subtrayectos_arco.py'")
s=s.replace("'tokens_subtrayectos.py',","'tokens_subtrayectos.py','tokens_subtrayectos_arco.py',")
s=s.replace("shutil.copy2(folder/'constructor_tokens.npz',run/'constructor_tokens.npz')",
 "shutil.copy2(OUT/'constructor_tokens_arco_s0/constructor_tokens.npz',run/'constructor_tokens.npz')")
s=s.replace("plan=read(OUT/'plan_atencion_subtrayectos.json')","plan=read(OUT/'plan_correccion_arco_subtrayectos.json')")
s=s.replace("write(run/'protocolo.json',config)",
 """write(run/'protocolo.json',config)""")
s=s.replace("if args.resume:assert read(run/'protocolo.json')==config",
 """best_reference=read(OUT/'nuevo_mejor_verificado.json');time_reference=read(OUT/'atencion_subtrayectos_s0/resultado.json')
config.update(geometry_sampling='Originalnormalizedmetricarclength and fullsourceintervallength',
 prefix_sha256=read(OUT/'distancia_arco_puntos_Train_s0/resultado.json')['outputs_sha256'],
 comparison_best_id=best_reference['id'],comparison_best_sha256=best_reference['model_sha256'],
 time_control_sha256=time_reference['model_sha256'])
if args.resume:assert read(run/'protocolo.json')==config""")
s=s.replace("model=SubtrajectoryAttention(parent,constructor['mean'],constructor['std'],constructor['time_anchor'],'individual').cuda()",
 """rng=torch.get_rng_state()
model=SubtrajectoryAttention(parent,constructor['mean'],constructor['std'],constructor['time_anchor'],'individual')
def initial_parameters_sha(network):
 digest=hashlib.sha256()
 for name,value in sorted(network.named_parameters()):
  if not name.startswith('parent.'):digest.update(name.encode());digest.update(value.detach().cpu().numpy().tobytes())
 return digest.hexdigest()
neural_sha=initial_parameters_sha(model)
torch.set_rng_state(rng)
old_ctor=dict(np.load(OUT/'atencion_subtrayectos_s0/constructor_tokens.npz'))
time_initial=SubtrajectoryAttention(parent,old_ctor['mean'],old_ctor['std'],old_ctor['time_anchor'],'individual')
assert initial_parameters_sha(time_initial)==neural_sha
del time_initial
write(run/'inicializacion_pareada.json',dict(passed=True,trainable_neural_parameter_sha256=neural_sha,
 identical_time_control_initial_neural_parameters=True,all_time_context_normalizers_exact=np.array_equal(constructor['mean'][16:],old_ctor['mean'][16:]) and np.array_equal(constructor['std'][16:],old_ctor['std'][16:]) and np.array_equal(constructor['time_anchor'],old_ctor['time_anchor'])))
model=model.cuda()""")
s=s.replace("library=SubtripTokenBank()","library=SubtripArcLengthBank()")
marker="x=torch.as_tensor(X,device='cuda');vx=torch.as_tensor(V,device='cuda');del X,V"
probe="""# Canonical inference preflight BEFORE allTrain tensors are resident.
probe_cpu=np.tile(np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index'][:32],64)
with np.load(OUT/'banco_proyeccion_incierta_s0/train_banco.npz') as z:probe_c=z['bank'][probe_cpu]
with np.load(OUT/'datos_prototipos_historicos_s0/train_historicos.npz') as z:probe_h=z['history'][probe_cpu]
with np.load(OUT/'datos_historia_individual_s0/train_indices.npz') as z:probe_old={key:torch.as_tensor(z[key][probe_cpu],device='cuda') for key in ['xy_km','hour','weekend','indices']}
with np.load(folder/'train_indices.npz') as z:probe_indices={key:z[key][probe_cpu] for key in ['parent','first','last']}
probe_sub={key:torch.as_tensor(value,device='cuda') for key,value in query_parts_sub({key:tr[key][probe_cpu] for key in ['o_lon','o_lat','d_lon','d_lat','timestamp']}).items()}
model.eval();torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
 probe_oraw=oldlibrary.features(probe_old['xy_km'],probe_old['hour'],probe_old['weekend'],probe_old['indices'])
 probe_sraw=library.features(probe_sub['xy_m'],probe_sub['hour'],probe_sub['weekend'],probe_indices['parent'],probe_indices['first'],probe_indices['last'])
 probe_out=model(torch.as_tensor(X[probe_cpu],device='cuda'),torch.as_tensor(probe_c,device='cuda'),torch.as_tensor(probe_h,device='cuda'),probe_oraw,probe_sraw)
 assert torch.isfinite(probe_out[0]).all()
 write(run/'preflight_consulta_2048.json',dict(passed=True,batch=2048,no_Validationmetric=True,GPU_reserved_MiB=torch.cuda.max_memory_reserved()/1024**2,all_Train_tensors_not_resident=True))
del probe_c,probe_h,probe_old,probe_indices,probe_sub,probe_oraw,probe_sraw,probe_out,probe_cpu
gc.collect();torch.cuda.empty_cache()
x=torch.as_tensor(X,device='cuda');vx=torch.as_tensor(V,device='cuda');del X,V"""
assert marker in s;s=s.replace(marker,probe)
s=s.replace("protected=predict(True);assert np.array_equal(protected[:,1],saved['pred'])\nwrite(run/'verificacion_referencia.json',dict(passed=True,all127941parent_points_exact=True,Test=False))",
 "assert read(OUT/'atencion_historia_individual_s0/certificacion/resultado.json')['passed']")
s=s.replace("model_digest=sha(path);p=predict();m=metric(truth,p[:,1]);comparison=paired(truth,p[:,1],saved['pred'],va['timestamp'])\ngate=comparison['delta_mae_s']<=-.5 and comparison['ci95_s'][1]<0",
 """model_digest=sha(path)
for container in [public,histories,oldqueries,subqueries,indices]:container[0]=None
del x,c,h,q,s,si,target,ids,g,valid,order,index,opt,loss
gc.collect();torch.cuda.empty_cache()
protected=predict(True);assert np.array_equal(protected[:,1],saved['pred'])
write(run/'verificacion_referencia.json',dict(passed=True,all127941parent_points_exact=True,Test=False))
p=predict();m=metric(truth,p[:,1]);comparison=paired(truth,p[:,1],saved['pred'],va['timestamp'])
time_saved=dict(np.load(OUT/'atencion_subtrayectos_s0/validacion.npz'));best_saved=dict(np.load(OUT/best_reference['id']/'validacion.npz'))
for z in [time_saved,best_saved]:
 for key in ['row','timestamp']:assert np.array_equal(z[key],va[key])
 assert np.array_equal(z['true'],truth)
vs_time=paired(truth,p[:,1],time_saved['pred'],va['timestamp']);vs_best=paired(truth,p[:,1],best_saved['pred'],va['timestamp'])
gate=all(v['delta_mae_s']<=-.5 and v['ci95_s'][1]<0 for v in [vs_time,vs_best])""")
s=s.replace("fixed_epoch=6,","fixed_epoch=6,paired_vs_time_sampling=vs_time,paired_vs_best=vs_best,")
s=s.replace("normalization_sha256=normal['constructor_sha256'],source_manifest_sha256=sha(folder/'resultado.json'),",
 "normalization_sha256=normal['constructor_sha256'],prefix_sha256=config['prefix_sha256'],source_manifest_sha256=sha(folder/'resultado.json'),")
s=s.replace("no Retained","no Retained")
assert 'SubtripTokenBank()' not in s and "constructor=dict(np.load(folder/'constructor_tokens.npz'))" not in s
assert "paired_vs_time_sampling=vs_time" in s and 'all_Train_tensors_not_resident=True' in s
ast.parse(s)
path=OUT/'entrenar_atencion_subtrayectos_arco.py';assert not path.exists();path.write_text(s)
write(OUT/'fuentes_entrenamiento_arco_preparadas.json',dict(prepared=True,not_launched=True,source_sha256=sha(path),
 original_time_trainer_sha256=sha(source),prefix_sha256=read(OUT/'distancia_arco_puntos_Train_s0/resultado.json')['outputs_sha256'],
 treatment_scope='Semanticgeometry/path correction only, 6fixedepochs, sameparent30 and sameinitialneuralparameters, alloriginalTrainqueries.',
 no_current_experiment_modified=True,Test='sealed'))
print('Separate6epoch arclength-correction trainer prepared and syntaxchecked; not launched.')
