"""CUDA throughput/memory feasibility, Train queries only; not a predictive evaluation."""
from base_replanteo import *
from modelo_dir_prior import PriorDirectionalArrival
from modelo_dir_fases import PhaseDirectionalArrival
from inferencia import feature_input
import torch,io,gc
torch.set_num_threads(4);torch.set_float32_matmul_precision('high');access=deny_test();assert (OUT/'espacial_direccional_s0/resultado.json').exists(),'Wait until current full inference finishes before adding GPU work'
parent=OUT/'espacial_prior_cf_s0'
for attempt in range(6):
 try:
  meta=read(parent/'mejor.json');st=(parent/'modelo.pt').stat();blob=(parent/'modelo.pt').read_bytes();st2=(parent/'modelo.pt').stat()
  if meta!=read(parent/'mejor.json') or st.st_mtime_ns!=st2.st_mtime_ns or len(blob)!=st.st_size:time.sleep(.2);continue
  ck=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False);break
 except (RuntimeError,OSError,EOFError):time.sleep(.2)
else:raise RuntimeError('No stable parent snapshot')
rec=read(BEST/'resultado.json');tr,X,names=feature_input('train',rec);y=np.load(DATA/'train_y.npy');grid=ck['grid'];graph=dict(np.load(OUT/'datos_semi_markov/grafo_grid.npz'));distance=np.load(OUT/'datos_semi_markov/distancias_grid.npy');counts=np.load(OUT/'prior_transiciones/conteos_fold.npz')['counts'];init=read(OUT/'inicializacion_fases.json')
snap_sha=hashlib.sha256(blob).hexdigest();models={'categorical':PriorDirectionalArrival(ck['stats'],grid,graph,distance,ck['power'],counts).cuda(),'phase':PhaseDirectionalArrival(ck['stats'],grid,graph,distance,ck['power'],counts,init).cuda()};models['categorical'].load_state_dict(ck['state']);shared=models['phase'].load_shared_state(ck['state']);del counts,ck,blob
use=np.flatnonzero((tr['timestamp']+y)<read(DATA/'manifiesto.json')['validation']['start_timestamp'])[:16]
def ids(tag):
 xx=(tr[tag+'_lon'][use]+8.62)*111195*np.cos(np.deg2rad(41.16));yy=(tr[tag+'_lat'][use]-41.16)*111195;return torch.tensor(np.floor((xx-grid['x0'])/200).astype(np.int64)*grid['ny']+np.floor((yy-grid['y0'])/200).astype(np.int64),device='cuda')
x=torch.tensor(X[use],device='cuda');label=torch.tensor(((y[use]-60)/15).astype(np.int64),device='cuda');origin=ids('o');destination=ids('d');fold=torch.tensor(((tr['timestamp'][use]//(7*86400))%5).astype(np.int64),device='cuda');del X
results=[]
for batch in [2,8,16]:
 for kind,model in models.items():
  for amp in ([False,True] if kind=='phase' else [False]):
   try:
    times=[];peaks=[]
    for repeat in range(3):
     model.eval();model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
     with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):p,mass,p0,path=model.posterior(x[:batch],origin[:batch],destination[:batch],excluded_fold=fold[:batch]);loss=-p.gather(1,label[:batch,None]).clamp_min(1e-9).log().mean()
     assert torch.isfinite(loss);loss.backward();torch.cuda.synchronize();elapsed=time.perf_counter()-start;assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None);times.append(elapsed);peaks.append(torch.cuda.max_memory_allocated()-baseline);del p,mass,p0,path,loss
    r=dict(kind=kind,batch=batch,AMP=amp,median_forward_backward_seconds=float(np.median(times)),Train_queries_per_second=batch/float(np.median(times)),incremental_peak_allocated_bytes=max(peaks),finite_gradients=True)
   except torch.cuda.OutOfMemoryError:
    model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();r=dict(kind=kind,batch=batch,AMP=amp,OOM=True)
   model.zero_grad(set_to_none=True);results.append(r);write(OUT/'benchmark_fases_cuda_progreso.json',results);print(json.dumps(r),flush=True)
report=dict(scope='Actual9,505directional states, Train-only query gradients, other empirical-prior job may be concurrent. Timings are feasibility measurements, not isolated hardware performance or MAE.',results=results,parent_checkpoint_snapshot_sha256=snap_sha,parent_selection_snapshot=meta,shared_initialization=shared,Test_access=False,code_sha256=sha(__file__))
write(OUT/'benchmark_fases_cuda.json',report)

