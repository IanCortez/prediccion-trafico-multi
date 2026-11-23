"""Raw65FIT-only GPU/CPU equivalentforward resourceproof; no queryT or MAE."""
from base_replanteo import *
from operador_red_cinetica import RoadClockKernel,RoadStateGeometry,load_counts
from operador_red_cinetica_GPU import CudaRoadClock
import torch,gc
start=time.time();access=deny_test();attempts=[]
forbidden=(OUT/'memoria_puntos_Train_s0/puntos_Train.npz').resolve()
def no_points(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
  try:p=Path(os.fsdecode(args[0])).resolve()
  except (OSError,ValueError):return
  if p==forbidden:attempts.append(str(p));raise AssertionError('Inferenceprobe cannot read originalpointbank')
sys.addaudithook(no_points)
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.use_deterministic_algorithms(True);torch.cuda.reset_peak_memory_stats()
run=OUT/'prueba_operador_cinetico_GPU_s0';assert not (run/'resultado.json').exists();run.mkdir(exist_ok=True)
parent=OUT/'prueba_operador_cinetico_Train_s0';proof=read(parent/'resultado.json');assert proof['passed']
saved=np.load(parent/'probabilidades_65FIT_sin_MAE.npz');assert sha(parent/'probabilidades_65FIT_sin_MAE.npz')==proof['data_sha256']
with np.load(DATA/'train_x.npz') as z:
 order=np.argsort(z['row']);chosen=order[np.searchsorted(z['row'][order],saved['row'])];assert np.array_equal(z['row'][chosen],saved['row'])
 query={key:z[key][chosen] for key in ['o_lon','o_lat','d_lon','d_lat','timestamp']}
fold=saved['query_fold'];counts=load_counts(OUT/'conteos_cineticos_muestra_Train_s0/conteos_fuentes4096.npz')
geometry=RoadStateGeometry(DATA/'mapa_dirigido_2014.npz');stats={};outputs={}
for kind in ['position','kinetic']:
 started=time.time();weights=np.zeros((65,116));reverse=np.zeros((17,116));cache=[]
 for excluded in np.unique(fold):
  ii=np.flatnonzero(fold==excluded);cpu=RoadClockKernel(counts,geometry,kind=kind,excluded_fold=int(excluded))
  model=CudaRoadClock(cpu);got=model.predict({key:value[ii] for key,value in query.items()})
  weights[ii]=got['weights'];cache.append(model.cache_peak)
  # Reverse17 +singleton in differentpartner blocks; all have canonical64innercolumns.
  original=np.arange(17)[::-1];original=original[fold[original]==excluded]
  other=model.predict({key:value[original] for key,value in query.items()})
  reverse[16-original]=other['weights']
  if fold[0]==excluded:
   single=model.predict({key:value[:1] for key,value in query.items()})
   assert np.array_equal(single['weights'][0],weights[0]),'Singleton exactinvariance'
  del model,cpu;gc.collect();torch.cuda.empty_cache()
 assert np.array_equal(reverse[::-1],weights[:17]),'Reverse exactinvariance'
 reference=saved[kind+'_weights'];mass=weights.sum(1);positive=mass>0
 probability=np.divide(weights,mass[:,None],out=np.zeros_like(weights),where=mass[:,None]>0)
 error=float(np.max(np.abs(weights-reference)))
 perr=float(np.max(np.abs(probability-saved[kind+'_probability'])))
 assert np.array_equal(positive,saved[kind+'_supported'])
 assert np.allclose(weights,reference,rtol=1e-10,atol=1e-14) and perr<1e-10,(error,perr)
 outputs[kind]=weights
 stats[kind]=dict(rawqueries=65,supported=int(positive.sum()),CPUmax_weight_error=error,CPUmax_probability_error=perr,
  reverse17_singleton_exact=True,cache_peak=max(cache),seconds=time.time()-started)
 print(json.dumps(dict(stage='CUDA_sparseforward_complete',kind=kind,result=stats[kind])),flush=True)
archive=run/'probabilidades_65FIT_GPU_sin_MAE.npz'
np.savez_compressed(archive,row=saved['row'],timestamp=query['timestamp'],**{kind+'_weights':value for kind,value in outputs.items()})
report=dict(passed=True,scope='GPUFloat64canonical64 sparseclock equivalence/resourceprobe. No predictiveMAE.',
 statistics=stats,GPU=torch.cuda.get_device_name(0),peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,
 peak_reserved_MiB=torch.cuda.max_memory_reserved()/1024**2,query_trueT_read=False,query_MAE_computed=False,
 originalGPSpointbank_open_attempts=attempts,all65_rows_retained=True,canonical_batch=64,kernel_cache_limit=4,
 CPUfulloperator_intervention_proof='verificacion_operador_cinetico_CPU_s0/resultado.json',
 production_scale_resourceproof_pending=True,Validation=False,Test=False,seconds=time.time()-start,
 actual_data_files_accessed=sorted(access),artifact_sha256=sha(archive),
 sources={name:sha(OUT/name) for name in ['probar_operador_cinetico_GPU.py','operador_red_cinetica_GPU.py','operador_red_cinetica.py']})
write(run/'resultado.json',report)
for name in report['sources']:(run/name).write_bytes((OUT/name).read_bytes())
print((run/'resultado.json').read_text(encoding='utf-8'),flush=True)
