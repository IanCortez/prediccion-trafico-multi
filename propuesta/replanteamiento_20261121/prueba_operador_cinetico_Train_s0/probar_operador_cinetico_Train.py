"""Raw65FIT-only probability/interface/resourceprobe; NEVER compare to actualqueryduration."""
from base_replanteo import *
from operador_red_cinetica import RoadClockKernel,RoadStateGeometry,load_counts
import ctypes
start=time.time();access=deny_test();attempts=[]
forbidden=(OUT/'memoria_puntos_Train_s0/puntos_Train.npz').resolve()
def no_points(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
  try:p=Path(os.fsdecode(args[0])).resolve()
  except (OSError,ValueError):return
  if p==forbidden:attempts.append(str(p));raise AssertionError('Inferenceprobe cannot read originalpointbank')
sys.addaudithook(no_points)
run=OUT/'prueba_operador_cinetico_Train_s0';assert not run.exists();run.mkdir()
assert read(OUT/'verificacion_operador_cinetico_CPU_s0/resultado.json')['passed']
folder=OUT/'conteos_cineticos_muestra_Train_s0';meta=read(folder/'resultado.json');assert meta['passed']
file=folder/'conteos_fuentes4096.npz';assert sha(file)==meta['artifact_sha256']
counts=load_counts(file)
with np.load(file) as z:parent=z['source_parent'];rows=z['source_row'];timestamp=z['source_timestamp']
chosen=np.linspace(0,len(parent)-1,65,dtype=int)
with np.load(DATA/'train_x.npz') as d:
 assert np.array_equal(d['row'][parent],rows) and np.array_equal(d['timestamp'][parent],timestamp)
 query={key:d[key][parent[chosen]] for key in ['o_lon','o_lat','d_lon','d_lat','timestamp']}
geometry=RoadStateGeometry(DATA/'mapa_dirigido_2014.npz')
fold=((query['timestamp']//604800)%5).astype(int)
stats={};predictions={};kernels={};times={}
for kind in ['position','kinetic']:
 outputs=dict(weights=np.zeros((65,116)),probability=np.zeros((65,116)),supported=np.zeros(65,bool),support=np.zeros(65))
 phase=time.time();nodes={};nonzero={};cache={}
 for excluded in np.unique(fold):
  ii=np.flatnonzero(fold==excluded)
  model=RoadClockKernel(counts,geometry,kind=kind,excluded_fold=int(excluded));kernels[kind,int(excluded)]=model
  sub={key:value[ii] for key,value in query.items()};result=model.predict(sub)
  for key in outputs:outputs[key][ii]=result[key]
  nodes[str(excluded)]=model.size;nonzero[str(excluded)]=model.global_P.nnz;cache[str(excluded)]=model.cache_peak
 elapsed=time.time()-phase;times[kind]=elapsed
 positive=outputs['supported'];normal_error=float(np.max(np.abs(outputs['probability'][positive].sum(1)-1),initial=0))
 assert np.isfinite(outputs['weights']).all() and (outputs['weights']>=0).all()
 assert normal_error<1e-12 and (outputs['support']<=1+1e-10).all()
 predictions[kind]=outputs
 # Repeat rawqueries in reversedorder across allsourcefolds; no privatequerylabel is read.
 reversed_weights=np.zeros((17,116))
 for j,original in enumerate(np.arange(17)[::-1]):
  q={key:value[original:original+1] for key,value in query.items()}
  model=kernels[kind,int(fold[original])];r=model.predict(q)
  reversed_weights[j]=r['weights'][0]
 assert np.array_equal(reversed_weights[::-1],outputs['weights'][:17])
 singleton=kernels[kind,int(fold[0])].predict({key:value[:1] for key,value in query.items()})
 assert np.array_equal(singleton['weights'][0],outputs['weights'][0])
 stats[kind]=dict(raw65_queries=65,supported_queries=int(positive.sum()),unsupported_rows_retained=int((~positive).sum()),
  normalization_maxerror=normal_error,max_support_mass=float(outputs['support'].max()),context_cache_peak=cache,
  compact_nodes_byfold=nodes,global_nonzero_transitions_byfold=nonzero,seconds=elapsed,
  reverse17_singleton_exact=True)
 print(json.dumps(dict(stage='sample_road_operator_complete',kind=kind,statistics=stats[kind])),flush=True)
assert not attempts
class MEMORY(ctypes.Structure):
 _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong),('PeakWorkingSetSize',ctypes.c_size_t),
 ('WorkingSetSize',ctypes.c_size_t),('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),
 ('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]
memory=MEMORY();memory.cb=ctypes.sizeof(memory)
ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(MEMORY),ctypes.c_ulong]
assert ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(memory),memory.cb)
archive=run/'probabilidades_65FIT_sin_MAE.npz'
np.savez_compressed(archive,row=rows[chosen],timestamp=query['timestamp'],query_fold=fold,
 **{kind+'_weights':p['weights'] for kind,p in predictions.items()},
 **{kind+'_probability':p['probability'] for kind,p in predictions.items()},
 **{kind+'_supported':p['supported'] for kind,p in predictions.items()})
report=dict(passed=True,scope='Raw65FITprobability/interface/resourceprobe only; no queryT or MAE.',
 results=stats,originalGPSpointbank_open_attempts=attempts,queryfields=list(query),
 sourcefold_removed_before_catalogue_backoff_andmeasurement=True,all65_rows_retained=True,
 unsupported_production_rule='Must substitute frozenreferencep0 before scoring; zeros here are explicitunsupportedflags, never a modelprediction.',
 peak_working_set_MiB=memory.PeakWorkingSetSize/1024**2,
 CPU_only=True,GPU_resourceprobe_pending=True,no_new_populationbank=True,
 data_sha256=sha(archive),sourcecounts_sha256=sha(file),actual_data_files_accessed=sorted(access),
 Validation=False,Test=False,query_trueT_read=False,query_MAE_computed=False,seconds=time.time()-start,
 sources={name:sha(OUT/name) for name in ['probar_operador_cinetico_Train.py','operador_red_cinetica.py','conteos_red_cinetica.py']})
write(run/'resultado.json',report)
for name in report['sources']:(run/name).write_bytes((OUT/name).read_bytes())
print((run/'resultado.json').read_text(encoding='utf-8'),flush=True)
