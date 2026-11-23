"""Fixed-six-epoch learned subtrajectory expert, retaining the protected parent hierarchy."""
from base_replanteo import *
import torch,shutil,gc,argparse
ap=argparse.ArgumentParser();ap.add_argument("--resume",action="store_true");args=ap.parse_args()
from inferencia import feature_input
from inferencia_atencion_historia import load_frozen_attention,raw_inputs
access=deny_test();start=time.time();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
torch.cuda.set_per_process_memory_fraction(.5);torch.manual_seed(20261122);torch.cuda.manual_seed_all(20261122)
parent,ck,reference,parent_run,proto,oldlibrary,helper,oldtoken=load_frozen_attention('atencion_historia_individual_s0',device='cpu')
from tokens_subtrayectos import query_parts_sub
from tokens_subtrayectos_arco import SubtripArcLengthBank
from modelo_atencion_subtrayectos import SubtrajectoryAttention
from modelo_aux_subviajes import duration_score
folder=OUT/'indices_subtrayectos_s0';meta=read(folder/'resultado.json');normal=read(OUT/'constructor_tokens_arco_s0/resultado.json')
assert meta['complete'] and normal['passed'] and sha(OUT/'constructor_tokens_arco_s0/constructor_tokens.npz')==normal['constructor_sha256']
assert normal['sources']['tokens_subtrayectos.py']==sha(OUT/'tokens_subtrayectos.py')
constructor=dict(np.load(OUT/'constructor_tokens_arco_s0/constructor_tokens.npz'))
run=OUT/'atencion_subtrayectos_arco_s0'
if not args.resume:
 assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
else:assert run.exists() and not (run/'resultado.json').exists()
sources=['entrenar_atencion_subtrayectos_arco.py','modelo_atencion_subtrayectos.py','tokens_subtrayectos.py','tokens_subtrayectos_arco.py',
 'inferencia_atencion_historia.py','base_replanteo.py','modelo_aux_subviajes.py','recuperar_subtrayectos.py','datos_progreso_recurrente.py']
for name in sources:
 if args.resume:assert sha(OUT/name)==sha(run/'fuentes'/name)
 else:shutil.copy2(OUT/name,run/'fuentes'/name)
if args.resume:assert sha(run/'constructor_tokens.npz')==normal['constructor_sha256']
else:shutil.copy2(OUT/'constructor_tokens_arco_s0/constructor_tokens.npz',run/'constructor_tokens.npz')
config=dict(epochs=6,batch=512,lr=.001,weight_decay=.0003,clip=5,seed=20261122,
 parent_id=reference['id'],parent_sha256=reference['model_sha256'],parent_validation=reference['validation'],
 plan=read(OUT/'plan_correccion_arco_subtrayectos.json'),selection='Fixedfinalepoch6 only, no Validationstopping or epochselection',
 source_manifest_sha256=sha(folder/'resultado.json'),normalization_sha256=normal['constructor_sha256'],
 GPU_fraction=.5,source_memory_sha256=meta['bank_sha256'],public_query=['o_lon','o_lat','d_lon','d_lat','timestamp'])
best_reference=read(OUT/'nuevo_mejor_verificado.json');time_reference=read(OUT/'atencion_subtrayectos_s0/resultado.json')
config.update(geometry_sampling='Originalnormalizedmetricarclength and fullsourceintervallength',
 prefix_sha256=read(OUT/'distancia_arco_puntos_Train_s0/resultado.json')['outputs_sha256'],
 comparison_best_id=best_reference['id'],comparison_best_sha256=best_reference['model_sha256'],
 time_control_sha256=time_reference['model_sha256'])
if args.resume:assert read(run/'protocolo.json')==config
else:write(run/'protocolo.json',config)
rng=torch.get_rng_state()
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
model=model.cuda()
library=SubtripArcLengthBank();tr,X,names=feature_input('train',read(BEST/'resultado.json'))
va,V,vnames=feature_input('validation',read(BEST/'resultado.json'));assert names==vnames==ck['features']
# Canonical inference preflight BEFORE allTrain tensors are resident.
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
x=torch.as_tensor(X,device='cuda');vx=torch.as_tensor(V,device='cuda');del X,V
public=[];histories=[];oldqueries=[];subqueries=[];indices=[]
for split,d in [('train',tr),('validation',va)]:
 cb=dict(np.load(OUT/'banco_proyeccion_incierta_s0'/(split+'_banco.npz')))
 hb=dict(np.load(OUT/'datos_prototipos_historicos_s0'/(split+'_historicos.npz')))
 oq=dict(np.load(OUT/'datos_historia_individual_s0'/(split+'_indices.npz')))
 sq=dict(np.load(folder/(split+'_indices.npz')))
 assert sha(folder/(split+'_indices.npz'))==meta['outputs'][split]['sha256']
 for key in ['row','timestamp']:
  for z in [cb,hb,oq,sq]:assert np.array_equal(d[key],z[key])
 public.append(torch.as_tensor(cb['bank'],device='cuda'));histories.append(torch.as_tensor(hb['history'],device='cuda'))
 oldqueries.append({key:torch.as_tensor(oq[key],device='cuda') for key in ['xy_km','hour','weekend','indices']})
 subqueries.append({key:torch.as_tensor(v,device='cuda') for key,v in query_parts_sub(d).items()})
 indices.append({key:torch.as_tensor(sq[key],device='cuda') for key in ['parent','first','last']})
 del cb,hb,oq,sq
c,vc=public;h,vh=histories;q,vq=oldqueries;s,vs=subqueries;si,vsi=indices
y=np.load(DATA/'train_y.npy');truth=np.load(DATA/'validation_y.npy');target=torch.as_tensor(y,device='cuda')
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index'];assert len(eligible)==1023485
assert np.max(tr['timestamp'][eligible]+y[eligible])<va['timestamp'].min();ids=torch.as_tensor(eligible,device='cuda')
gt=dict(np.load(DATA/'geometria_train.npz'));pos=np.searchsorted(gt['row'],tr['row']);safe=np.minimum(pos,len(gt['row'])-1)
valid=(pos<len(gt['row']))&(gt['row'][safe]==tr['row'])&(gt['X'][safe,-1]<=500)
geometry=gt['X'][safe,1:15].copy();valid&=np.isfinite(geometry).all(1);geometry[~valid]=0
g=torch.as_tensor(geometry,device='cuda');valid=torch.as_tensor(valid,device='cuda',dtype=torch.float32);del gt,geometry,pos,safe
saved=dict(np.load(parent_run/'validacion.npz'))
for key in ['row','timestamp']:assert np.array_equal(saved[key],va[key])
assert np.array_equal(saved['true'],truth)
def tokens(query,subquery,indexbank,index):
 old=oldlibrary.features(query['xy_km'][index],query['hour'][index],query['weekend'][index],query['indices'][index])
 sub=library.features(subquery['xy_m'][index],subquery['hour'][index],subquery['weekend'][index],
  indexbank['parent'][index],indexbank['first'][index],indexbank['last'][index])
 return old,sub
def objective(index):
 old,sub=tokens(q,s,si,index);out=model(x[index],c[index],h[index],old,sub);t=target[index]
 return duration_score(out[0],t,model.duration.duration_grid)+.2*model.duration.geometry_score(out,t,g[index],valid[index])
# Actual full data resident before GPU preflight; do not extrapolate from a tiny mock.
for size in [32,512]:
 model.train();torch.cuda.reset_peak_memory_stats()
 tick=time.time();loss=objective(ids[:size]);assert torch.isfinite(loss);loss.backward();torch.cuda.synchronize()
 assert all(p.grad is None for p in model.parent.parameters())
 assert all(torch.isfinite(p.grad).all() for p in model.branch_parameters() if p.grad is not None)
 peak=torch.cuda.max_memory_reserved()/1024**2
 model.zero_grad(set_to_none=True)
 write(run/('preflight_'+str(size)+'.json'),dict(passed=True,batch=size,loss=float(loss.detach()),GPU_reserved_MiB=peak,seconds=time.time()-tick,parent_gradients_absent=True))
@torch.no_grad()
def predict(parent_only=False):
 model.eval();parts=[]
 for first in range(0,len(vx),2048):
  n=min(2048,len(vx)-first);index=torch.arange(first,first+n,device='cuda')
  if n<2048:index=torch.cat([index,index[-1:].expand(2048-n)])
  old,sub=tokens(vq,vs,vsi,index)
  out=model.parent(vx[index],vc[index],vh[index],old) if parent_only else model(vx[index],vc[index],vh[index],old,sub)
  parts.append(model.duration.quantiles(out[0])[:n].cpu().numpy());del old,sub,out
 return np.concatenate(parts)
assert read(OUT/'atencion_historia_individual_s0/certificacion/resultado.json')['passed']
opt=torch.optim.AdamW(model.branch_parameters(),lr=.001,weight_decay=.0003);history=[];first_epoch=1
if args.resume:
 recovery=torch.load(run/'ultima_reanudacion.pt',map_location='cpu',weights_only=False)
 assert recovery['config']==config
 check=model.load_state_dict(recovery['branch_state'],strict=False)
 assert not check.unexpected_keys and all(k.startswith('parent.') for k in check.missing_keys)
 opt.load_state_dict(recovery['optimizer']);history=recovery['history'];first_epoch=recovery['epoch']+1
 torch.set_rng_state(recovery['torch_rng']);torch.cuda.set_rng_state_all(recovery['cuda_rng'])
for epoch in range(first_epoch,7):
 model.train();tick=time.time();total=0.;count=0
 order=ids[torch.randperm(len(ids),device='cuda',generator=torch.Generator(device='cuda').manual_seed(1000+epoch))]
 for index in order.split(512):
  opt.zero_grad(set_to_none=True);loss=objective(index);assert torch.isfinite(loss);loss.backward()
  torch.nn.utils.clip_grad_norm_(model.branch_parameters(),5,error_if_nonfinite=True);opt.step()
  total+=float(loss.detach())*len(index);count+=len(index)
 assert count==1023485
 history.append(dict(epoch=epoch,primary_train_n=count,train_loss=total/count,seconds=time.time()-tick))
 write(run/'historia.json',history);print(json.dumps(history[-1]),flush=True)
 # Only the latest small branch state for interruption recovery; no duplicate parentweights.
 branch={key:value for key,value in model.state_dict().items() if not key.startswith('parent.')}
 torch.save(dict(branch_state=branch,optimizer=opt.state_dict(),epoch=epoch,config=config,history=history,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),run/'ultima_reanudacion.pt')
branch={key:value for key,value in model.state_dict().items() if not key.startswith('parent.')}
path=run/'epoca_6.pt';torch.save(dict(branch_state=branch,constructor=constructor,config=config,features=names,epoch=6),path)
model_digest=sha(path)
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
gate=all(v['delta_mae_s']<=-.5 and v['ci95_s'][1]<0 for v in [vs_time,vs_best])
# Public raw5 source retrieval and tokens agree exactly with cached candidates.
from recuperar_subtrayectos import SubtrajectoryMemory
ii=np.sort(np.random.default_rng(20261122).choice(len(truth),65,replace=False));fields=config['public_query']
items=[{k:int(va[k][i]) if k=='timestamp' else float(va[k][i]) for k in fields} for i in ii]
memory=SubtrajectoryMemory();memory.build_tree(5)
@torch.no_grad()
def raw_predict(items):
 X,RC,RH,RQ,rnames=raw_inputs(items,ck['base_config'],proto,oldlibrary,helper,oldtoken);assert rnames==names
 dd={k:np.array([v[k] for v in items]) for k in fields};source=memory.query(dd)
 st={k:torch.as_tensor(v,device='cuda') for k,v in query_parts_sub(dd).items()}
 raw=library.features(st['xy_m'],st['hour'],st['weekend'],source[0],source[1],source[2])
 index=torch.arange(len(items),device='cuda')
 # Exact canonical padding to 2048 for parent and new head.
 padded=torch.cat([index,index[-1:].expand(2048-len(items))])
 tq={k:torch.as_tensor(v,device='cuda')[padded] for k,v in RQ.items()}
 old=oldlibrary.features(tq['xy_km'],tq['hour'],tq['weekend'],tq['indices'])
 arrays=[torch.as_tensor(v,device='cuda')[padded] for v in [X,RC,RH]]
 out=model(*arrays,old,raw[padded]);pred=model.duration.quantiles(out[0])[:len(items)].cpu().numpy()
 return pred,raw.cpu().numpy(),source
raw,rawtokens,rs=raw_predict(items);reverse,rev_tokens,_=raw_predict(items[::-1]);single,single_tokens,_=raw_predict(items[:1])
assert np.array_equal(raw,p[ii]) and np.array_equal(raw,reverse[::-1]) and np.array_equal(raw[:1],single)
for name,source in zip(['parent','first','last'],rs[:3]):assert np.array_equal(source,indices[1][name][ii].cpu().numpy())
_,cached=tokens(vq,vs,vsi,torch.as_tensor(ii,device='cuda'));assert np.array_equal(rawtokens,cached.cpu().numpy())
assert np.array_equal(rawtokens,rev_tokens[::-1]) and np.array_equal(rawtokens[:1],single_tokens)
np.savez_compressed(run/'validacion.npz',row=va['row'],timestamp=va['timestamp'],true=truth,pred=p[:,1],quantiles=p,reference=saved['pred'])
qualified=m['mae_s']<131.06 or m['mae_s']<reference['validation']['mae_s']
result=dict(complete=True,id=run.name,config=config,validation=m,paired_vs_parent=comparison,meaningful_gate_passed=bool(gate),
 qualified=bool(qualified),model_file=str(path) if qualified else None,model_sha256=model_digest,fixed_epoch=6,paired_vs_time_sampling=vs_time,paired_vs_best=vs_best,
 parent_unchanged=sha(reference['model_file'])==reference['model_sha256'],parent_source=reference['id'],
 raw65_cached_sources_tokens_full_quantiles_exact=True,raw_reverse_singleton_exact=True,
 all_original_primary_queries_and_T=True,Train_parent_source_fold_excluded=True,
 normalization_sha256=normal['constructor_sha256'],prefix_sha256=config['prefix_sha256'],source_manifest_sha256=sha(folder/'resultado.json'),
 sources={name:sha(run/'fuentes'/name) for name in sources},predictions_sha256=sha(run/'validacion.npz'),
 uncertainty=dict(coverage80=float(np.mean((truth>=p[:,0])&(truth<=p[:,2]))),width80_s=float((p[:,2]-p[:,0]).mean())),
 Test_access=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(run/'resultado.json',result);write(run/'consultas_raw_sin_etiquetas.json',items)
cleanup=[]
if not qualified:
 assert path.resolve().is_relative_to(run.resolve()) and sha(path)==model_digest;path.unlink();cleanup.append(dict(file=path.name,sha256=model_digest))
p=run/'ultima_reanudacion.pt';assert p.resolve().is_relative_to(run.resolve());digest=sha(p);p.unlink();cleanup.append(dict(file=p.name,sha256=digest))
write(run/'limpieza.json',dict(removed=cleanup,completed_metrics_predictions_sources_preserved=True))
print(json.dumps(read(run/'resultado.json')),flush=True)
