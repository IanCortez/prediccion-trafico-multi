"""Recover completed training's Validation inference only; no training or Test access."""
from base_replanteo import *
from modelo_dir_prior import PriorDirectionalArrival
from inferencia import feature_input
import torch,argparse
ap=argparse.ArgumentParser();ap.add_argument('--probe-only',action='store_true');a=ap.parse_args();access=deny_test();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction(.25,0);run=OUT/'espacial_prior_cf_s0';model_hash=sha(run/'modelo.pt');ck=torch.load(run/'modelo.pt',map_location='cpu',weights_only=False);grid=ck['grid'];graph=dict(np.load(OUT/'datos_semi_markov/grafo_grid.npz'));distance=np.load(OUT/'datos_semi_markov/distancias_grid.npy');counts=np.load(OUT/'prior_transiciones/conteos_fold.npz')['counts'];model=PriorDirectionalArrival(ck['stats'],grid,graph,distance,ck['power'],counts);model.load_state_dict(ck['state']);del counts,ck
# These large excluded-fold buffers are never read at Validation, where excluded_fold=None.
for name in ['counts_fold','counts_base_fold','counts_global_fold']:setattr(model,name,torch.empty(0))
model=model.cuda().eval();rec=read(BEST/'resultado.json');va,V,names=feature_input('validation',rec);assert names==rec['features'];vx=torch.tensor(V,device='cuda');del V
def ids(tag):
 xx=(va[tag+'_lon']+8.62)*111195*np.cos(np.deg2rad(41.16));yy=(va[tag+'_lat']-41.16)*111195;aa=np.floor((xx-grid['x0'])/200).astype(np.int64);bb=np.floor((yy-grid['y0'])/200).astype(np.int64);assert np.all((aa>=0)&(aa<grid['nx'])&(bb>=0)&(bb<grid['ny']));return torch.tensor(aa*grid['ny']+bb,device='cuda')
origin=ids('o');destination=ids('d');N=len(va['row']);pred=np.full(N,np.nan,np.float32);pod=pred.copy();pg=pred.copy();mass=pred.copy();progress=run/'progreso_validacion.npz';start=time.time()
if progress.exists():
 z=np.load(progress);assert np.array_equal(z['row'],va['row']) and str(z['model_sha256'])==model_hash;pred=z['pred'].copy();pod=z['od_head'].copy();pg=z['graph'].copy();mass=z['mass'].copy();z.close()
@torch.no_grad()
def block(first):
 ix=torch.arange(first,min(first+8,N),device='cuda');n=len(ix)
 if n<8:ix=torch.cat([ix,ix[-1:].expand(8-n)])
 p,m,p0,path=model.posterior(vx[ix],origin[ix],destination[ix]);assert torch.isfinite(p).all() and torch.isfinite(m).all();out=model.median(p);od=model.median(p0);g=torch.where(m>1e-6,model.median(path),od);return [z[:n].cpu().numpy() for z in [out,od,g,m]]
def save_progress():
 np.savez_compressed(progress,row=va['row'],pred=pred,od_head=pod,graph=pg,mass=mass,model_sha256=np.array(model_hash))
 write(run/'recuperacion_estado.json',dict(pid=os.getpid(),model_sha256=model_hash,completed_rows=int(np.isfinite(pred).sum()),total_rows=N,seconds=time.time()-start,GPU_memory_fraction=.25,canonical_batch=8,process='inference only',Test_access=False))
# First probe the interval around the previous failure, preserving original canonical batches.
if a.probe_only:
 for first in range(12288,14336,8):
  out=block(first);last=first+len(out[0]);pred[first:last],pod[first:last],pg[first:last],mass[first:last]=out
 save_progress();write(run/'sondeo_recuperacion.json',dict(passed=True,interval=[12288,14336],rows=2048,CUDA_LAUNCH_BLOCKING=os.environ.get('CUDA_LAUNCH_BLOCKING'),model_sha256=model_hash,seconds=time.time()-start,Test_access=False));print('Failure interval passed in a single bounded GPU process.',flush=True);exit()
for first in range(0,N,8):
 last=min(first+8,N)
 if not np.isfinite(pred[first:last]).all():
  out=block(first);pred[first:last],pod[first:last],pg[first:last],mass[first:last]=out
 if first%4096==0:
  save_progress();print(json.dumps(dict(stage='recovered_full_validation_prediction',completed_rows=int(np.isfinite(pred).sum()),total_rows=N,seconds=time.time()-start)),flush=True)
assert np.isfinite(pred).all();assert sha(run/'modelo.pt')==model_hash;v=np.load(DATA/'validation_y.npy');old=np.load(BEST/'validacion.npz');np.savez_compressed(run/'validacion.npz',row=va['row'],timestamp=va['timestamp'],true=v,pred=pred,od_head=pod,graph_with_support_fallback=pg,absorption_mass=mass)
result=dict(id=run.name,config=read(run/'protocolo.json'),validation=metric(v,pred),od_validation=metric(v,pod),graph_validation=metric(v,pg),best_stage=read(run/'mejor.json')['stage'],model_sha256=model_hash,model_code_sha256=sha(OUT/'modelo_dir_prior.py'),kernel_code_sha256=sha(OUT/'nucleo_direccional.py'),training_code_sha256=sha(run/'fuentes/entrenar_dir_prior.py'),inference_recovery_code_sha256=sha(__file__),prior_counts_sha256=sha(OUT/'prior_transiciones/conteos_fold.npz'),data_grid_sha256=sha(OUT/'datos_semi_markov/resultado.json'),paired_vs_retained=paired(v,pred,old['pred'],va['timestamp']),parameters=sum(p.numel() for p in model.parameters()),recovered_after_cuda_failure=True,training_was_not_repeated=True,canonical_batch=8,unused_fold_buffers_removed_only_from_RAM=True,Test_access=False,goal_validation=metric(v,pred)['mae_s']<73.5,actual_data_files_accessed=sorted(access),recovery_seconds=time.time()-start)
write(run/'resultado.json',result);progress.unlink();print(json.dumps({k:result[k] for k in ['id','validation','od_validation','graph_validation','recovery_seconds']}),flush=True)

