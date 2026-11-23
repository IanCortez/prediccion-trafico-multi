"""Fixed24epoch jointlytrained fresh30component OD/time/source mixture; no pretrainedneuralweights."""
from base_replanteo import *
import torch,shutil,gc,argparse
ap=argparse.ArgumentParser();ap.add_argument("--resume",action="store_true");args=ap.parse_args()
from inferencia import feature_input
from inferencia_historica_fresca import raw_inputs
from modelo_mezcla_rutas import make_duration_model
from modelo_mezcla_proyeccion import ProjectionMixture
from modelo_mezcla_historica import HistoricalRouteMixture
from historicos_rutas import HistoryRouteBank
access=deny_test();start=time.time();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
torch.cuda.set_per_process_memory_fraction(.5);torch.manual_seed(20261122);torch.cuda.manual_seed_all(20261122)
treatment=read(OUT/'subtrayectos_conjuntos_s0/resultado.json')
assert treatment['complete'] and treatment['meaningful_gate_passed'],'Controlnotauthorizedwithoutpredeclaredusefulgain'
previous=read(OUT/'atencion_subtrayectos_s0/resultado.json');assert previous['complete']
reference=read(OUT/treatment['config']['comparison_reference_id']/'resultado.json')
assert reference['model_sha256']==treatment['config']['comparison_reference_sha256']
parent_run=OUT/reference['id']
base_ctor=torch.load(OUT/'historica_fresca_s0/constructor_base.pt',map_location='cpu',weights_only=False)
assert 'state' not in base_ctor and base_ctor['Train_n']==1023485
hctor=dict(np.load(OUT/'historica_fresca_s0/constructor_historia.npz'))
mctor=dict(np.load(OUT/'historica_fresca_s0/constructor_mapa.npz'))
base_record=read(BEST/'resultado.json')
ck=dict(features=base_ctor['features'],base_config=base_record['config'])
duration=make_duration_model(base_ctor,base_record['config'])
projection=ProjectionMixture(duration,mctor['mean'],mctor['std'],'expanded')
parent=HistoricalRouteMixture(projection,hctor['mean'],hctor['std'],hctor['time_anchor'],'historical')
with torch.no_grad():parent.gate[-1].bias.zero_()
proto=HistoryRouteBank(OUT/'datos_prototipos_historicos_s0/memoria_Train.npz')
from tokens_subtrayectos import SubtripTokenBank,query_parts_sub
from modelo_subtrayectos_conjuntos import JointSubtrajectoryAttention
from modelo_aux_subviajes import duration_score
folder=OUT/'indices_subtrayectos_s0';meta=read(folder/'resultado.json');normal=read(folder/'normalizacion_resultado.json')
assert meta['complete'] and normal['passed'] and sha(folder/'constructor_tokens.npz')==normal['constructor_sha256']
assert normal['sources']['tokens_subtrayectos.py']==sha(OUT/'tokens_subtrayectos.py')
constructor=dict(np.load(folder/'constructor_tokens.npz'))
run=OUT/'control_subtrayectos_conjuntos_s0'
if not args.resume:
 assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
else:assert run.exists() and not (run/'resultado.json').exists()
sources=['entrenar_control_subtrayectos_conjuntos.py','modelo_subtrayectos_conjuntos.py','tokens_subtrayectos.py',
 'inferencia_historica_fresca.py','modelo_mezcla_proyeccion.py','modelo_mezcla_historica.py','historicos_rutas.py','piloto_prototipos_ruta.py','base_replanteo.py','modelo_aux_subviajes.py','recuperar_subtrayectos.py','datos_progreso_recurrente.py']
for name in sources:
 if args.resume:assert sha(OUT/name)==sha(run/'fuentes'/name)
 else:shutil.copy2(OUT/name,run/'fuentes'/name)
if args.resume:assert sha(run/'constructor_tokens.npz')==normal['constructor_sha256']
else:shutil.copy2(folder/'constructor_tokens.npz',run/'constructor_tokens.npz')
config=dict(mode='matched_source_value_control',treatment_id='subtrayectos_conjuntos_s0',treatment_result_sha256=sha(OUT/'subtrayectos_conjuntos_s0/resultado.json'),epochs=24,batch=512,lr=.001,weight_decay=.0003,clip=5,seed=20261122,
 comparison_reference_id=reference['id'],comparison_reference_sha256=reference['model_sha256'],reference_validation=reference['validation'],
 constructor_sha256={name:sha(OUT/'historica_fresca_s0'/name) for name in ['constructor_base.pt','constructor_historia.npz','constructor_mapa.npz']},
 plan=read(OUT/'plan_subtrayectos_conjuntos.json'),selection='Allfreshneuralweights, fixedfinalepoch24, no Validationstopping or epochselection',previous_result_sha256=sha(OUT/'atencion_subtrayectos_s0/resultado.json'),
 source_manifest_sha256=sha(folder/'resultado.json'),normalization_sha256=normal['constructor_sha256'],
 GPU_fraction=.5,source_memory_sha256=meta['bank_sha256'],public_query=['o_lon','o_lat','d_lon','d_lat','timestamp'])
if args.resume:assert read(run/'protocolo.json')==config
else:write(run/'protocolo.json',config)
geometry_proof=read(OUT/'verificacion_geometria_tokens_subtrayectos.json')
assert geometry_proof['passed'] and geometry_proof['token_sha256']==sha(OUT/'tokens_subtrayectos.py')
model=JointSubtrajectoryAttention(parent,constructor['mean'],constructor['std'],constructor['time_anchor'],'individual')
initial=hashlib.sha256()
for name,value in sorted(model.state_dict().items()):initial.update(name.encode());initial.update(value.detach().numpy().tobytes())
assert initial.hexdigest()==read(OUT/'subtrayectos_conjuntos_s0/inicializacion_fresca.json')['initial_state_sha256']
write(run/'inicializacion_fresca.json',dict(initial_state_sha256=initial.hexdigest(),all_neural_parameters_require_grad=all(p.requires_grad for p in model.parameters()),neural_pretrained_weights_loaded=False))
model=model.cuda()
library=SubtripTokenBank();tr,X,names=feature_input('train',read(BEST/'resultado.json'))
va,V,vnames=feature_input('validation',read(BEST/'resultado.json'));assert names==vnames==ck['features']
x=torch.as_tensor(X,device='cuda');vx=torch.as_tensor(V,device='cuda');del X,V
public=[];histories=[];subqueries=[];indices=[]
for split,d in [('train',tr),('validation',va)]:
 cb=dict(np.load(OUT/'banco_proyeccion_incierta_s0'/(split+'_banco.npz')))
 hb=dict(np.load(OUT/'datos_prototipos_historicos_s0'/(split+'_historicos.npz')))
 sq=dict(np.load(folder/(split+'_indices.npz')))
 assert sha(folder/(split+'_indices.npz'))==meta['outputs'][split]['sha256']
 for key in ['row','timestamp']:
  for z in [cb,hb,sq]:assert np.array_equal(d[key],z[key])
 public.append(torch.as_tensor(cb['bank'],device='cuda'));histories.append(torch.as_tensor(hb['history'],device='cuda'))
 subqueries.append({key:torch.as_tensor(v,device='cuda') for key,v in query_parts_sub(d).items()})
 indices.append({key:torch.as_tensor(sq[key],device='cuda') for key in ['parent','first','last']})
 del cb,hb,sq
c,vc=public;h,vh=histories;s,vs=subqueries;si,vsi=indices
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
def remove_source_values(raw):
 # Keep source availability, endpoints, localclock, start/stopfractions and rankingmetadata.
 # Remove measuredgeometry/path/sourceelapsedtime AND full-parentlogduration (column32),
 # otherwise fullParentT times intervalfraction would reconstructsourceelapsedtime.
 raw=raw.clone();raw[...,:17]=model.token_mean[:17].to(raw.dtype);raw[...,32]=model.token_mean[32].to(raw.dtype)
 return raw
def tokens(subquery,indexbank,index):
 sub=library.features(subquery['xy_m'][index],subquery['hour'][index],subquery['weekend'][index],
  indexbank['parent'][index],indexbank['first'][index],indexbank['last'][index])
 return remove_source_values(sub)
def objective(index):
 sub=tokens(s,si,index);out=model(x[index],c[index],h[index],sub);t=target[index]
 return duration_score(out[0],t,model.duration.duration_grid)+.2*model.duration.geometry_score(out,t,g[index],valid[index])
# Actual full data resident before GPU preflight; do not extrapolate from a tiny mock.
for size in [32,512]:
 model.train();torch.cuda.reset_peak_memory_stats()
 tick=time.time();loss=objective(ids[:size]);assert torch.isfinite(loss);loss.backward();torch.cuda.synchronize()
 assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 assert any(p.grad is not None and p.grad.norm()>0 for p in model.duration.parameters())
 assert model.head[-1].weight.grad is not None and model.head[-1].weight.grad.norm()>0
 assert all(p.requires_grad for p in model.parameters())
 assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 peak=torch.cuda.max_memory_reserved()/1024**2
 model.zero_grad(set_to_none=True)
 write(run/('preflight_'+str(size)+'.json'),dict(passed=True,batch=size,loss=float(loss.detach()),GPU_reserved_MiB=peak,seconds=time.time()-tick,fresh_parent_joint_gradients_finite=True))
# Verify the actual canonical inference memory peak before committing to training.
model.eval()
with torch.no_grad():
 probe=ids[:32].repeat(64);torch.cuda.reset_peak_memory_stats();sub=tokens(s,si,probe)
 output=model(x[probe],c[probe],h[probe],sub)
 assert torch.isfinite(output[0]).all()
 write(run/'preflight_consulta_2048.json',dict(passed=True,batch=2048,GPU_reserved_MiB=torch.cuda.max_memory_reserved()/1024**2,no_Validationmetric=True))
 del sub,output,probe
@torch.no_grad()
def predict():
 model.eval();parts=[]
 for first in range(0,len(vx),2048):
  n=min(2048,len(vx)-first);index=torch.arange(first,first+n,device='cuda')
  if n<2048:index=torch.cat([index,index[-1:].expand(2048-n)])
  sub=tokens(vs,vsi,index)
  out=model(vx[index],vc[index],vh[index],sub)
  parts.append(model.duration.quantiles(out[0])[:n].cpu().numpy());del sub,out
 return np.concatenate(parts)
write(run/'verificacion_inicializacion.json',dict(passed=True,no_retained_network_state_loaded=True,base_constructor_sha256=sha(OUT/'historica_fresca_s0/constructor_base.pt'),protected_reference_unchanged=sha(reference['model_file'])==reference['model_sha256'],Test=False))
opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0003);history=[];first_epoch=1
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=24,eta_min=1e-5)
if args.resume:
 recovery=torch.load(run/'ultima_reanudacion.pt',map_location='cpu',weights_only=False)
 assert recovery['config']==config
 model.load_state_dict(recovery['state'],strict=True)
 scheduler.load_state_dict(recovery['scheduler'])
 opt.load_state_dict(recovery['optimizer']);history=recovery['history'];first_epoch=recovery['epoch']+1
 torch.set_rng_state(recovery['torch_rng']);torch.cuda.set_rng_state_all(recovery['cuda_rng'])
for epoch in range(first_epoch,25):
 model.train();tick=time.time();total=0.;count=0
 order=ids[torch.randperm(len(ids),device='cuda',generator=torch.Generator(device='cuda').manual_seed(1000+epoch))]
 for index in order.split(512):
  opt.zero_grad(set_to_none=True);loss=objective(index);assert torch.isfinite(loss);loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
  total+=float(loss.detach())*len(index);count+=len(index)
 assert count==1023485
 scheduler.step()
 history.append(dict(epoch=epoch,primary_train_n=count,train_loss=total/count,lr=opt.param_groups[0]['lr'],seconds=time.time()-tick))
 write(run/'historia.json',history);print(json.dumps(history[-1]),flush=True)
 # Keep one full model+optimizer checkpoint for interruption recovery.
 branch=model.state_dict()
 torch.save(dict(state=branch,optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,config=config,history=history,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),run/'ultima_reanudacion.pt')
branch=model.state_dict()
path=run/'epoca_24.pt';torch.save(dict(state=branch,constructor=constructor,base_constructor=base_ctor,base_config=base_record['config'],history_constructor=hctor,map_constructor=mctor,config=config,features=names,epoch=24),path)
model_digest=sha(path)
# Release Train tensors before full evaluation; Validation queries and source memory suffice.
for container in [public,histories,subqueries,indices]:container[0]=None
del x,c,h,s,si,target,ids,g,valid,order,index,opt,loss
gc.collect();torch.cuda.empty_cache()
p=predict();m=metric(truth,p[:,1]);comparison=paired(truth,p[:,1],saved['pred'],va['timestamp'])
gate=comparison['delta_mae_s']<=-.5 and comparison['ci95_s'][1]<0
# Public raw5 source retrieval and tokens agree exactly with cached candidates.
from recuperar_subtrayectos import SubtrajectoryMemory
ii=np.sort(np.random.default_rng(20261122).choice(len(truth),65,replace=False));fields=config['public_query']
items=[{k:int(va[k][i]) if k=='timestamp' else float(va[k][i]) for k in fields} for i in ii]
memory=SubtrajectoryMemory();memory.build_tree(5)
@torch.no_grad()
def raw_predict(items):
 X,RC,RH,rnames=raw_inputs(items,ck['base_config'],proto);assert rnames==names
 dd={k:np.array([v[k] for v in items]) for k in fields};source=memory.query(dd)
 st={k:torch.as_tensor(v,device='cuda') for k,v in query_parts_sub(dd).items()}
 raw=remove_source_values(library.features(st['xy_m'],st['hour'],st['weekend'],source[0],source[1],source[2]))
 index=torch.arange(len(items),device='cuda')
 # Exact canonical padding to 2048 for parent and new head.
 padded=torch.cat([index,index[-1:].expand(2048-len(items))])
 arrays=[torch.as_tensor(v,device='cuda')[padded] for v in [X,RC,RH]]
 out=model(*arrays,raw[padded]);pred=model.duration.quantiles(out[0])[:len(items)].cpu().numpy()
 return pred,raw.cpu().numpy(),source
raw,rawtokens,rs=raw_predict(items);reverse,rev_tokens,_=raw_predict(items[::-1]);single,single_tokens,_=raw_predict(items[:1])
assert np.array_equal(raw,p[ii]) and np.array_equal(raw,reverse[::-1]) and np.array_equal(raw[:1],single)
for name,source in zip(['parent','first','last'],rs[:3]):assert np.array_equal(source,indices[1][name][ii].cpu().numpy())
cached=tokens(vs,vsi,torch.as_tensor(ii,device='cuda'));assert np.array_equal(rawtokens,cached.cpu().numpy())
assert np.array_equal(rawtokens,rev_tokens[::-1]) and np.array_equal(rawtokens[:1],single_tokens)
np.savez_compressed(run/'validacion.npz',row=va['row'],timestamp=va['timestamp'],true=truth,pred=p[:,1],quantiles=p,reference=saved['pred'])
qualified=m['mae_s']<131.06 or m['mae_s']<reference['validation']['mae_s']
result=dict(complete=True,id=run.name,config=config,validation=m,paired_vs_parent=comparison,meaningful_gate_passed=bool(gate),
 qualified=bool(qualified),model_file=str(path) if qualified else None,model_sha256=model_digest,fixed_epoch=24,
 protected_reference_unchanged=sha(reference['model_file'])==reference['model_sha256'],comparison_reference=reference['id'],trained_from_scratch=True,
 raw65_cached_sources_tokens_full_quantiles_exact=True,raw_reverse_singleton_exact=True,
 all_original_primary_queries_and_T=True,Train_parent_source_fold_excluded=True,
 normalization_sha256=normal['constructor_sha256'],source_manifest_sha256=sha(folder/'resultado.json'),
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
