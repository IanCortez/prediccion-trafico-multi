"""One predeclared global blend calibration after the physical model is fully frozen."""
from base_replanteo import *
from modelo_dir_fases import PhaseDirectionalArrival
from modelo_mezcla_rutas import make_duration_model
from inferencia import feature_input
import torch
torch.set_num_threads(4)
torch.set_float32_matmul_precision('highest')
access=deny_test()
parent=OUT/'espacial_fases_referencia_s0'
assert (parent/'resultado.json').exists(),'Do not run while training is active'
torch.cuda.set_per_process_memory_fraction(.25,0)
run=OUT/'calibracion_unica_fases_s0'
assert not run.exists()
run.mkdir()
protocol=read(OUT/'protocolo_calibracion_unica.json')
alphas=protocol['alphas']
r=read(parent/'resultado.json')
assert sha(parent/'modelo.pt')==r['model_sha256']
ck=torch.load(parent/'modelo.pt',map_location='cpu',weights_only=False)
grid=ck['grid']
graph=dict(np.load(OUT/'datos_semi_markov/grafo_grid.npz'))
distance=np.load(OUT/'datos_semi_markov/distancias_grid.npy')
counts=np.load(OUT/'prior_transiciones/conteos_fold.npz')['counts']
model=PhaseDirectionalArrival(ck['stats'],grid,graph,distance,ck['power'],counts,ck['phase_initialization'])
model.load_state_dict(ck['state'])
for name in ['counts_fold','counts_base_fold','counts_global_fold']:setattr(model,name,torch.empty(0))
model=model.cuda().eval()
del ck,counts
rec=read(BEST/'resultado.json')
va,V,names=feature_input('validation',rec)
vx=torch.tensor(V,device='cuda')
del V
old=np.load(BEST/'validacion.npz')
y=old['true']
assert np.array_equal(old['row'],va['row'])
def ids(tag):
 xx=(va[tag+'_lon']+8.62)*111195*np.cos(np.deg2rad(41.16))
 yy=(va[tag+'_lat']-41.16)*111195
 return torch.tensor(np.floor((xx-grid['x0'])/200).astype(np.int64)*grid['ny']+np.floor((yy-grid['y0'])/200).astype(np.int64),device='cuda')
origin,dest=ids('o'),ids('d')
original=torch.load(BEST/'modelo.pt',map_location='cpu',weights_only=False)
teacher=make_duration_model(original,rec['config']).cuda().eval()
teacher.load_state_dict(original['state'])
reference=torch.empty((len(vx),116),device='cuda')
with torch.no_grad():
 for first in range(0,len(vx),2048):
  block=vx[first:first+2048];n=len(block)
  if n<2048:block=torch.cat([block,block[-1:].expand(2048-n,-1)])
  reference[first:first+n]=teacher(block)[0].exp().mean(1)[:n]
assert np.array_equal(model.median(reference).cpu().numpy(),old['pred'])
del teacher,original
torch.cuda.empty_cache()
proxy=np.load(parent/'selection_indices.npy')
@torch.no_grad()
def predict(indices):
 out=np.empty((len(alphas),len(indices)),np.float32)
 for first in range(0,len(indices),32):
  ii=torch.tensor(indices[first:first+32],device='cuda');n=len(ii)
  if n<32:ii=torch.cat([ii,ii[-1:].expand(32-n)])
  p,mass,ref,path=model.posterior(vx[ii],origin[ii],dest[ii],reference_probability=reference[ii])
  support=mass/(mass+.01)
  for j,alpha in enumerate(alphas):
   if alpha==0:mix=ref
   else:
    weight=alpha*support
    mix=(1-weight[:,None])*ref+weight[:,None]*path
    mix=mix/mix.sum(-1,keepdim=True)
   out[j,first:first+n]=model.median(mix)[:n].cpu().numpy()
  if len(indices)>4096 and first%4096==0:print(json.dumps(dict(stage='one_blend_scan',completed_rows=first,total_rows=len(indices))),flush=True)
 return out
selection=predict(proxy)
selection_metrics=[metric(y[proxy],p) for p in selection]
selected=int(np.argmin([m['mae_s'] for m in selection_metrics]))
write(run/'seleccion.json',dict(alphas=alphas,metrics=selection_metrics,chosen_alpha=alphas[selected],criterion='same fixed4096 selection rows; tie favors smaller weight',not_full_cohort=True))
pred=predict(np.arange(len(y)))
assert np.array_equal(pred[0],old['pred'])
full=[metric(y,p) for p in pred]
unique=[];mapping=[]
for p in pred:
 if np.array_equal(p,old['pred']):mapping.append(-1);continue
 found=next((i for i,q in enumerate(unique) if np.array_equal(p,q)),None)
 if found is None:found=len(unique);unique.append(p.copy())
 mapping.append(found)
np.savez_compressed(run/'predicciones_distintas.npz',row=va['row'],timestamp=va['timestamp'],true=y,pred=np.stack(unique) if unique else np.empty((0,len(y))),alpha_to_row=np.array(mapping),alphas=np.array(alphas))
oracle_tested=np.abs(pred.astype(np.float64)-y.astype(np.float64)[None,:]).min(0)
qualified=[]
for alpha,mm in zip(alphas,full):
 if mm['mae_s']<131.06:
  item=dict(alpha=alpha,validation=mm,shared_checkpoint=str(parent/'modelo.pt'),checkpoint_sha256=r['model_sha256'],reference_sha256=rec['model_sha256'],output='median of (1-alpha*support)*protectedCDF +alpha*support*physicalCDF; support=mass/(mass+.01)',canonical_batch=32,reference_batch=2048,no_duplicate_weights=True,Test_access=False)
  qualified.append(item)
write(run/'candidatos_a_conservar.json',qualified)
result=dict(protocol=protocol,alphas=alphas,full_validation=full,selected_by_fixed_subset=alphas[selected],selected_full_validation=full[selected],best_full_validation_exploratory_alpha=alphas[int(np.argmin([m['mae_s'] for m in full]))],shared_checkpoint_sha256=r['model_sha256'],reference_sha256=rec['model_sha256'],paired_selected_vs_reference=paired(y,pred[selected],old['pred'],va['timestamp']),ORACLE_min_error_among_only_the8_tested_outputs_s=float(oracle_tested.mean()),oracle_scope='Privileged per-query choice among only the8 actual predeclared outputs, using unknown truth. Not deployable, not achieved MAE, and not a lower bound for untested weights or other architectures. The single-scan limit comes from the predeclared protocol.',qualified_candidates=qualified,actual_data_files_accessed=sorted(access),Test_access=False,code_sha256=sha(__file__))
write(run/'resultado.json',result)
print(json.dumps({k:result[k] for k in ['alphas','full_validation','selected_by_fixed_subset','qualified_candidates','ORACLE_min_error_among_only_the8_tested_outputs_s']}),flush=True)

