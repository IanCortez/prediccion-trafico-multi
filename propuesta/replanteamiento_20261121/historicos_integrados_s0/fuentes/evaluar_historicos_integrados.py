"""Frozen-reference historical-estimator integration, fixed policies, full cohort."""
from base_replanteo import *
import torch,argparse,shutil
from inferencia import feature_input
from consulta_historicos_cruzados import prepare,history_features,MOVEMENT_NAMES,STATIC_NAMES
from modelo_mezcla_rutas import make_duration_model
ap=argparse.ArgumentParser();ap.add_argument('--id',default='historicos_integrados_s0');args=ap.parse_args()
access=deny_test();proof=read(OUT/'verificacion_historicos_cruzados.json')
assert proof['passed'] and proof['source_sha256']==sha(OUT/'consulta_historicos_cruzados.py')
run=OUT/args.id;assert run.resolve().parent==OUT.resolve() and not run.exists();run.mkdir();(run/'fuentes').mkdir()
start=time.time();plan=dict(scope='Primary unchanged full Validation; frozen original duration model',
 hypothesis='Training features use whole-week-excluded historical estimates; integrate the same five available estimators at query instead of a single full-Train point estimate.',
 main_policy='Uniform average of five predicted duration probability vectors, held-history-fold0..4, weights fixed1/5. No fitting or weight search.',
 controls=['Original full-Train profile5, must exactly reproduce all protected point predictions','Single profile indexed by unchanged raw UTC departure week modulo5, matching Train feature-construction rule'],
 query=['o_lon','o_lat','d_lon','d_lat','timestamp'],unchanged='Original labels, filters, cohort,216feature definitions, neural weights, static routes and timestamp',
 changed='Only10 historical movement feature values for two already selected static routes; all remaining206 features and road geometry stay fixed.',
 historical_data='Already frozen Train-only profiles, own block removed at every backoff. This is a finite jackknife-style sensitivity integration, not a claim of exact Bayesian posterior draws.',
 precision='FP32 highest, no autocast, canonical2048, no probability renormalization beyond original model',
 loss_or_optimizer='None; no parameter updates',Validation_selection='Policies fixed before calculation; no individual-view MAE/selection',Test='sealed',
 resources='CPU2 preprocessing, one GPU inference job, six small movement arrays in RAM only, no disk feature or CDF banks',
 conservation='Preserve each eligible policy below131.06 and any new best. Original checkpoint is shared rather than duplicated. Raw five-field proof already passed on65.')
write(run/'plan.json',plan)
for name in ['evaluar_historicos_integrados.py','consulta_historicos_cruzados.py','base_replanteo.py']:shutil.copy2(OUT/name,run/'fuentes'/name)
write(run/'fuentes/parent_sources.json',{str(p.relative_to(FROZEN)):sha(p) for p in FROZEN.rglob('*.py')})
rec=read(BEST/'resultado.json');assert sha(BEST/'modelo.pt')==rec['model_sha256']
va,V,names=feature_input('validation',rec);assert names==rec['features']
protected=np.load(BEST/'validacion.npz');reference_pred=protected['pred'];assert np.array_equal(protected['row'],va['row']) and len(V)==127941
mi=[names.index(n) for n in MOVEMENT_NAMES];si=[names.index(n) for n in STATIC_NAMES]
columns=['o_lon','o_lat','d_lon','d_lat','timestamp']
items=[{k:int(va[k][i]) if k=='timestamp' else float(va[k][i]) for k in columns} for i in range(len(V))]
prepared=prepare(items);del items
M5,S5=history_features(prepared,5);m_error=float(np.abs(M5-V[:,mi]).max());s_error=float(np.abs(S5-V[:,si]).max())
assert m_error<1e-5 and s_error<1e-5,(m_error,s_error,'Raw control mismatch')
views={}
for held_fold in range(5):
 M,S=history_features(prepared,held_fold);assert np.array_equal(S,S5);views[held_fold]=M
 print(json.dumps(dict(stage='historical_view_ready',held_fold=held_fold,rows=len(M),static_features_unchanged=True,seconds=time.time()-start)),flush=True)
profile_meta=read(DATA/'perfiles_mapmatched_2014/manifiesto.json')
profile_hashes={name:sha(DATA/'perfiles_mapmatched_2014'/name) for name in ['time_s.npy','state_support.npy','link_support.npy']}
assert all(profile_hashes[name]==profile_meta['outputs'][name] for name in profile_hashes)
# Freeze all query feature/policy construction before opening duration labels.
del prepared,S,S5
torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction(.4)
ck=torch.load(BEST/'modelo.pt',map_location='cpu',weights_only=False);model=make_duration_model(ck,rec['config']).cuda();model.load_state_dict(ck['state']);model.eval()
for p in model.parameters():p.requires_grad_(False)
grid=model.duration_grid
def median(prob):return grid[(prob.cumsum(-1)<.5).sum(-1).clamp_max(115)]
pred={k:np.empty(len(V),np.float32) for k in ['full_history_control','calendar_fold_control','integrated_five_views']}
@torch.no_grad()
def predict_block(features,view_arrays,weekfold):
 x=torch.tensor(features,device='cuda');base=model(x)[0].exp().mean(1)
 summed=torch.zeros_like(base);calendar=torch.zeros_like(base);assigned=torch.zeros(len(x),device='cuda',dtype=torch.int32)
 wf=torch.tensor(weekfold,device='cuda');columns=torch.tensor(mi,device='cuda')
 for fold in range(5):
  modified=x.clone();modified[:,columns]=torch.tensor(view_arrays[fold],device='cuda')
  p=model(modified)[0].exp().mean(1);summed+=p
  mask=wf==fold;calendar[mask]=p[mask];assigned+=mask.int()
 assert torch.all(assigned==1)
 avg=summed/5
 assert torch.isfinite(avg).all() and (avg.sum(-1)-1).abs().max()<1e-5
 return dict(full_history_control=median(base).cpu().numpy(),calendar_fold_control=median(calendar).cpu().numpy(),integrated_five_views=median(avg).cpu().numpy())
weekfold=(va['timestamp']//(7*86400))%5
for first in range(0,len(V),2048):
 last=min(first+2048,len(V));block=V[first:last].copy();block[:,mi]=M5[first:last]
 result=predict_block(block,{f:views[f][first:last] for f in range(5)},weekfold[first:last])
 for key in pred:pred[key][first:last]=result[key]
 if first%16384==0:print(json.dumps(dict(stage='full_profile_policy_inference',rows=last,total=len(V),seconds=time.time()-start)),flush=True)
assert np.array_equal(pred['full_history_control'],reference_pred),'Original full-history numerical control failed'
# Point-query order and singleton checks on frozen policies.
ix=np.arange(65)[::-1];block=V[ix].copy();block[:,mi]=M5[ix]
reverse=predict_block(block,{f:views[f][ix] for f in range(5)},weekfold[ix])
checks={}
for name in pred:checks[name]=dict(reverse65_equal=bool(np.array_equal(reverse[name],pred[name][ix])))
for index in [0,1,63,64]:
 block=V[index:index+1].copy();block[:,mi]=M5[index:index+1]
 one=predict_block(block,{f:views[f][index:index+1] for f in range(5)},weekfold[index:index+1])
 for name in pred:
  checks[name].setdefault('singleton4_equal',True)
  checks[name]['singleton4_equal'] &= bool(np.array_equal(one[name],pred[name][index:index+1]))
y=np.load(DATA/'validation_y.npy');assert np.array_equal(y,protected['true'])
outputs={name:dict(validation=metric(y,p),paired_vs_reference=paired(y,p,reference_pred,va['timestamp']),raw_order_checks=checks[name]) for name,p in pred.items()}
np.savez_compressed(run/'validacion.npz',row=va['row'],timestamp=va['timestamp'],true=y,**pred)
qualified=[]
for name in ['calendar_fold_control','integrated_five_views']:
 m=outputs[name]['validation']
 if m['mae_s']<131.06 or m['mae_s']<rec['validation']['mae_s']:
  policy=dict(name=name,main_policy=name=='integrated_five_views',reference_model=str((BEST/'modelo.pt').relative_to(ROOT)),reference_sha256=rec['model_sha256'],history_folds=[0,1,2,3,4],weights=[.2]*5 if name=='integrated_five_views' else 'Query UTCweek modulo5 picks one estimator',features=names,profiles_sha256=profile_hashes,precision=plan['precision'],query=columns,source_sha256=sha(OUT/'consulta_historicos_cruzados.py'),validation=m,raw_reproduction_pending=True)
  write(run/(name+'_politica.json'),policy);qualified.append(dict(name=name,validation=m,policy_file=name+'_politica.json',threshold_qualifies=m['mae_s']<131.06,raw_reproduction_pending=True))
record=dict(id=args.id,query_information=columns,primary_Validation_n=len(y),outputs=outputs,qualified_candidates=qualified,
 main_policy='integrated_five_views',full_history_control_exact=True,full_raw_movement_feature_max_error=m_error,full_raw_static_feature_max_error=s_error,
 original_model_unchanged=sha(BEST/'modelo.pt')==rec['model_sha256'],reference_sha256=rec['model_sha256'],profiles_sha256=profile_hashes,
 sources={p.name:sha(p) for p in (run/'fuentes').glob('*.py')},predictions_sha256=sha(run/'validacion.npz'),Test_access=False,actual_files_accessed=sorted(access),elapsed_seconds=time.time()-start,
 conclusion='Compare fixed integration policy with original full-history and matched-construction control; no individual-view or blending-weight selection. A failed result does not complete the research goal.',
 goal='active')
write(run/'resultado.json',record);write(OUT/'conclusion_historicos_integrados.json',record)
print(json.dumps(dict(id=args.id,outputs=outputs,qualified_candidates=qualified,seconds=time.time()-start),default=lambda v:v.tolist() if isinstance(v,np.ndarray) else str(v)),flush=True)
