"""Road-time observations before chronological Train cutoff, tagged by whole parent week."""
from base_replanteo import *
import shutil
access=deny_test();start=time.time();run=OUT/'datos_trafico_tensor_train_s0';assert not run.exists();run.mkdir();shutil.copy2(__file__,run/'fuente_ejecutada.py')
d=dict(np.load(DATA/'train_x.npz'));eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index'];split=int(d['timestamp'][eligible].max())-21*86400
corpus=OUT/'datos_calles_parciales_Train_s0';meta=read(corpus/'resultado.json');assert meta['complete'];rng=np.random.default_rng(20261122)
parts={key:[] for key in ['chain','distance_m','duration_s','entry_timestamp','parent_fold','parent_row','train_index']}
for item in meta['shards']:
 path=corpus/item['file'];assert sha(path)==item['sha256'];z=dict(np.load(path))
 chosen=[];entry=[];parent=[]
 for i,(a,b) in enumerate(zip(z['offsets'][:-1],z['offsets'][1:])):
  if z['timestamp'][i]+z['true_duration_s'][i]>=split:continue
  known=np.flatnonzero((z['chain'][a:b]!=65535)&(z['duration_s'][a:b]>0))+a
  if not len(known):continue
  ix=rng.choice(known,min(2,len(known)),replace=False);prefix=np.r_[0,np.cumsum(z['duration_s'][a:b],dtype=np.float64)]
  chosen.extend(ix.tolist());entry.extend(np.round(z['timestamp'][i]+prefix[ix-a],3).tolist());parent.extend([i]*len(ix))
 if not chosen:continue
 chosen=np.asarray(chosen,np.int64);parent=np.asarray(parent,np.int64)
 for key,zkey in [('chain','chain'),('distance_m','distance_m'),('duration_s','duration_s')]:parts[key].append(z[zkey][chosen])
 parts['entry_timestamp'].append(np.asarray(entry,np.float64))
 for key,zkey in [('parent_fold','fold'),('parent_row','row'),('train_index','train_index')]:parts[key].append(z[zkey][parent])
arrays={key:np.concatenate(value) for key,value in parts.items()};n=len(arrays['chain'])
if n>1000000:
 keep=np.sort(rng.choice(n,1000000,replace=False));arrays={key:value[keep] for key,value in arrays.items()}
assert np.max(arrays['entry_timestamp'])<split and np.isfinite(arrays['duration_s']).all() and (arrays['duration_s']>0).all()
assert np.all((arrays['parent_fold']>=0)&(arrays['parent_fold']<5))
file=run/'observaciones_Train.npz';np.savez_compressed(file,**arrays)
result=dict(complete=True,observations=len(arrays['chain']),available_observations=n,seed=20261122,max_per_parent_trip=2,max_observations=1000000,
 chronological_cutoff=split,all_source_arrivals_before_cutoff=True,parent_fold='ParentqueryUTCweek modulo5; entire source trip is excludedforcrossfit, includingallsubsegments.',
 local_clock='ObservedTrainroadentry derived fromsum-preservingpartialtimes, rounded1millisecond to avoid numericalboundaryjitter; not query input. Predictionusesquerydeparture/publicfreeflowarrival.',
 observations_sha256=sha(file),source_sha256=sha(run/'fuente_ejecutada.py'),corpus_manifest_sha256=sha(corpus/'resultado.json'),
 artifact_MiB=file.stat().st_size/1024**2,Validation_access=False,Test_access=False,seconds=time.time()-start)
write(run/'resultado.json',result);print(json.dumps(result),flush=True)
