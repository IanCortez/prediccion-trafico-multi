"""Fullsourcegraph raw5 predictions on the frozen50kqueries; no queryT/MAE access."""
from base_replanteo import *
import torch,gc,pandas as pd
from operador_red_cinetica import RoadClockKernel,RoadStateGeometry,load_counts
from operador_red_cinetica_GPU import CudaRoadClock
from inferencia import feature_input
from modelo_control_lognormal_puente import PublicLognormalControl,canonical_predict_control
from datos_mezcla_individual_Train import build_public206
start=time.time();access=deny_test();attempts=[]
forbidden=(OUT/'memoria_puntos_Train_s0/puntos_Train.npz').resolve()
def no_points(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
  try:p=Path(os.fsdecode(args[0])).resolve()
  except (OSError,ValueError):return
  if p==forbidden:attempts.append(str(p));raise AssertionError('Graphqueryinference cannot open originalpointbank')
sys.addaudithook(no_points)
protocol=read(OUT/'protocolo_red_cinetica_Train.json');assert protocol['status']=='FROZEN_BEFORE_GRAPH_TIME_METRIC'
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True);torch.cuda.set_per_process_memory_fraction(.5)
torch.cuda.reset_peak_memory_stats()
run=OUT/'indices_red_cinetica_Train_s0';run.mkdir(exist_ok=True);assert not (run/'resultado.json').exists()
sources={name:sha(OUT/name) for name in ['preparar_predicciones_red_cinetica_Train.py','operador_red_cinetica.py','operador_red_cinetica_GPU.py','conteos_red_cinetica.py','modelo_control_lognormal_puente.py']}
if (run/'sources.json').exists():assert read(run/'sources.json')==sources
else:
 write(run/'sources.json',sources)
 for name in sources:(run/name).write_bytes((OUT/name).read_bytes())
 write(run/'protocolo.json',protocol)
with np.load(OUT/'consultas_red_cinetica_Train.npz') as z:
 index=z['train_index'];rows=z['row'];timestamp=z['timestamp'];nfit=int(z['fit_n'])
assert nfit==40000 and len(index)==50000 and sha(OUT/'consultas_red_cinetica_Train.npz')==protocol['query_metadata_sha256']
d,X,names=feature_input('train',read(BEST/'resultado.json'));keep=np.array([not name.startswith('mm_') for name in names])
X=np.ascontiguousarray(X[index][:,keep]);names=[name for name,k in zip(names,keep) if k]
assert X.shape==(50000,206) and np.isfinite(X).all() and np.array_equal(d['row'][index],rows) and np.array_equal(d['timestamp'][index],timestamp)
query={key:d[key][index] for key in ['o_lon','o_lat','d_lon','d_lat','timestamp']};del d
probe=np.linspace(0,len(index)-1,65,dtype=int)
items=[{key:int(value[i]) if key=='timestamp' else float(value[i]) for key,value in query.items()} for i in probe]
raw,names2=build_public206(items,read(BEST/'resultado.json')['config']);assert names==names2 and np.array_equal(raw,X[probe])
reference=read(OUT/'referencia_publica_Train_red_vial.json');file=Path(reference['model_file'])
assert sha(file)==protocol['reference_model_sha256']==reference['model_sha256']
saved=torch.load(file,map_location='cpu',weights_only=False);model=PublicLognormalControl(**saved['constructor']).cuda();model.load_state_dict(saved['state']);model.eval()
lp,q=canonical_predict_control(model,torch.as_tensor(X,device='cuda'));p0=lp.exp().numpy()
assert np.isfinite(p0).all() and np.max(np.abs(p0.sum(1)-1))<1e-12
with np.load(OUT/'control_familia_puente_2d_Train_s0/predicciones_held_Train.npz') as z:
 assert np.array_equal(rows[nfit:],z['row']) and np.array_equal(timestamp[nfit:],z['timestamp'])
 assert np.array_equal(q[nfit:].numpy(),z['quantiles']) and np.array_equal(lp[nfit:].numpy().astype(np.float32),z['log_prob'])
del model,saved,lp,q;gc.collect();torch.cuda.empty_cache()
source=OUT/'conteos_cineticos_prefijo_Train_s0';proof=read(source/'resultado.json');assert proof['passed']
countfile=source/'conteos_fuentes.npz';assert sha(countfile)==proof['artifact_sha256']
counts=load_counts(countfile);geometry=RoadStateGeometry(DATA/'mapa_dirigido_2014.npz')
fold=((timestamp//604800)%5).astype(int);fold[nfit:]=5
outprob={};outsupport={};outmass={};statistics={}
for kind in ['position','kinetic']:
 probability=np.zeros((50000,116),np.float32);supported=np.zeros(50000,bool);support=np.zeros(50000,np.float64);stats=[]
 for excluded in range(6):
  ii=np.flatnonzero(fold==excluded);path=run/(kind+'_fold'+str(excluded)+'.npz');meta=path.with_suffix('.json')
  if path.exists():
   item=read(meta);assert sha(path)==item['sha256'] and item['sourcecounts_sha256']==proof['artifact_sha256']
   with np.load(path) as z:
    assert np.array_equal(z['indices'],ii);probability[ii]=z['probability'];supported[ii]=z['supported'];support[ii]=z['support']
   stats.append(item);continue
  phase=time.time();cpu=RoadClockKernel(counts,geometry,kind=kind,excluded_fold=excluded);model=CudaRoadClock(cpu)
  print(json.dumps(dict(stage='fullsource_kernel_ready',kind=kind,excluded_fold=excluded,query_n=len(ii),compact_nodes=cpu.size,global_nnz=cpu.global_P.nnz,seconds=time.time()-start)),flush=True)
  if excluded==0:
   # Fixedproduction resourcecheck chosen solely by publicdeparturehour frequency, before anyquerylabel.
   local=pd.to_datetime(timestamp[ii],unit='s',utc=True).tz_convert('Europe/Lisbon')
   how=np.asarray(local.dayofweek)*24+np.asarray(local.hour);values,num=np.unique(how,return_counts=True)
   jj=ii[np.flatnonzero(how==values[np.argmax(num)])[:64]]
   small={key:value[jj] for key,value in query.items()};gpu=model.predict(small)
   test=cpu.predict({key:value[jj[:3]] for key,value in query.items()})
   assert np.allclose(gpu['weights'][:3],test['weights'],rtol=1e-10,atol=1e-14)
   assert np.max(np.abs(gpu['probability'][:3]-test['probability']),initial=0)<1e-10
   rev=model.predict({key:value[jj[:17][::-1]] for key,value in query.items()})
   single=model.predict({key:value[jj[:1]] for key,value in query.items()})
   assert np.max(np.abs(rev['probability'][::-1]-gpu['probability'][:17]),initial=0)<1e-10
   assert np.max(np.abs(single['probability']-gpu['probability'][:1]),initial=0)<1e-10
   resource=dict(passed=True,queries=len(jj),compact_nodes=cpu.size,global_nnz=cpu.global_P.nnz,
    CPU_first3_max_probability_error=float(np.max(np.abs(gpu['probability'][:3]-test['probability']),initial=0)),
    reverse17_singleton_max_probability_error=float(max(np.max(np.abs(rev['probability'][::-1]-gpu['probability'][:17]),initial=0),np.max(np.abs(single['probability']-gpu['probability'][:1]),initial=0))),
    peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,peak_reserved_MiB=torch.cuda.max_memory_reserved()/1024**2,
    seconds=time.time()-phase,no_queryT_read=True,no_MAE=True)
   assert resource['peak_allocated_MiB']<5500
   write(run/(kind+'_production_resourceproof.json'),resource);print(json.dumps(dict(stage='production_resourceproof',kind=kind,resource=resource)),flush=True)
  lastprint=[0.]
  def progress(detail):
   now=time.time()
   if now-lastprint[0]>30:
    lastprint[0]=now;print(json.dumps(dict(stage='graph_forward',kind=kind,excluded_fold=excluded,**detail,seconds=now-start)),flush=True)
  result=model.predict({key:value[ii] for key,value in query.items()},progress)
  probability[ii]=result['probability'].astype(np.float32);supported[ii]=result['supported'];support[ii]=result['support']
  np.savez_compressed(path,indices=ii,probability=probability[ii],supported=supported[ii],support=support[ii])
  item=dict(kind=kind,excluded_fold=excluded,queries=len(ii),supported=int(supported[ii].sum()),compact_nodes=cpu.size,
   global_nnz=cpu.global_P.nnz,seconds=time.time()-phase,sourcecounts_sha256=proof['artifact_sha256'],sha256=sha(path))
  write(meta,item);stats.append(item)
  print(json.dumps(dict(stage='graph_fold_complete',**item,total_seconds=time.time()-start)),flush=True)
  del model,cpu,result;gc.collect();torch.cuda.empty_cache()
 outprob[kind]=probability;outsupport[kind]=supported;outmass[kind]=support;statistics[kind]=stats
archive=run/'consultas_predicciones.npz'
np.savez_compressed(archive,train_index=index,row=rows,timestamp=timestamp,fit_n=np.array(nfit),query_fold=fold,
 publicX=X,reference_probability=p0,
 **{kind+'_probability':value for kind,value in outprob.items()},
 **{kind+'_supported':value for kind,value in outsupport.items()},
 **{kind+'_support':value for kind,value in outmass.items()})
# Sourcecounts plus predictionfeatures and stillpresentfoldshards obey same350MiB hardcap.
footprint=sum(p.stat().st_size for folder in [source,run] for p in folder.rglob('*') if p.is_file())
assert footprint<350*1024**2
journal=[]
with np.load(archive) as z:
 for kind in ['position','kinetic']:
  assert np.array_equal(z[kind+'_probability'],outprob[kind]) and np.array_equal(z[kind+'_supported'],outsupport[kind])
for kind in ['position','kinetic']:
 for excluded in range(6):
  path=run/(kind+'_fold'+str(excluded)+'.npz');journal.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size))
write(run/'retiro_shards_predicciones.json',dict(merged_sha256=sha(archive),files=journal,reason='Exactmergedprobabilityarchive verified; temporaryfoldcopies only.'))
for item in journal:
 path=ROOT/item['path'];assert path.resolve().is_relative_to(run.resolve()) and sha(path)==item['sha256'];path.unlink()
report=dict(passed=True,fitqueries=nfit,heldqueries=10000,sourceprefix_n=proof['fit_sourceparents'],statistics=statistics,
 graph_support={kind:dict(fit=int(outsupport[kind][:nfit].sum()),held=int(outsupport[kind][nfit:].sum())) for kind in outprob},
 reference_exact_replayed_onheld=True,raw65_public206_exact=True,queryfields=list(query),original_pointbank_open_attempts=attempts,
 no_actualqueryroad_speed_N_T_or_path=True,queryT_read=False,query_MAE_computed=False,wholeparentfold_removed_before_allkernelconstruction=True,
 helduses_all_past_fit_only=True,all50000_originalrows_retained=True,unsupported_rule='Finalmixture exactlyp0; never removequery.',
 artifact_sha256=sha(archive),artifact_MiB=archive.stat().st_size/1024**2,sourcecounts_sha256=proof['artifact_sha256'],
 peak_persistent_counts_plusfeatures_MiB=footprint/1024**2,peak_GPU_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,
 seconds=time.time()-start,Validation=False,Test=False,actual_data_files_accessed=sorted(access),sources=sources)
write(run/'resultado.json',report);print((run/'resultado.json').read_text(encoding='utf-8'),flush=True)
