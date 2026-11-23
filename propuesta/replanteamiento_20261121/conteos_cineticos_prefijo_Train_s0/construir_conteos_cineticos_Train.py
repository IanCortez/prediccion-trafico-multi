"""Bounded/resumable951718FIT sourcecounts; never persist populationpoint/roadlabels."""
from base_replanteo import *
import pandas as pd,gc
from estados_cineticos_15s import PublicRoadSampler,fitted_pair_distances
from conteos_red_cinetica import *
CAP=350*1024**2;BATCH=4096
def encode_batch(state,valid,error,offsets,timestamp,K):
 n=np.diff(offsets);p=np.repeat(np.arange(len(n)),n)
 step=np.arange(len(state))-np.repeat(offsets[:-1],n)
 clock=timestamp[p]+15*step
 local=pd.to_datetime(clock,unit='s',utc=True).tz_convert('Europe/Lisbon')
 how=np.asarray(local.dayofweek,dtype=np.int16)*24+np.asarray(local.hour,dtype=np.int16)
 context=(((timestamp//604800)%5).astype(np.int16)[p]*168+how)
 ii=np.flatnonzero(valid)
 assert np.all(p[ii]==p[ii+1]) and np.all(clock[ii+1]-clock[ii]==15)
 k,c=aggregate_keys(encode_pairs(context[ii],state[ii],state[ii+1],K))
 end=offsets[1:]-1;end=end[error[end]<=80]
 begin=offsets[:-1];begin=begin[(error[begin]<=80)&valid[begin]]
 sk,sc=aggregate_keys(encode_events(context[end],state[end],K))
 ak,ac=aggregate_keys(encode_events(context[begin],state[begin],K))
 return dict(kinetic_pair_keys=k,kinetic_pair_counts=c,stop_keys=sk,stop_counts=sc,start_keys=ak,start_counts=ac)
def footprint(run):
 return sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
class BudgetFile:
 def __init__(self,raw,cap):self.raw=raw;self.cap=cap;self.high=0
 def write(self,data):
  high=max(self.high,self.raw.tell()+len(data))
  if high>self.cap:raise RuntimeError('350MiB sourcecounts/features hardbudget exceeded before write')
  result=self.raw.write(data);self.high=high;return result
 def __getattr__(self,name):return getattr(self.raw,name)
def bounded_save(path,run,**arrays):
 assert not path.exists()
 remaining=CAP-footprint(run)
 try:
  with path.open('w+b') as f:np.savez_compressed(BudgetFile(f,remaining),**arrays)
 except BaseException:
  if path.exists():
   write(run/'incomplete_write_retirement.json',dict(path=str(path),bytes=path.stat().st_size,reason='Failedboundedwrite; completedshards unchanged'))
   assert path.resolve().is_relative_to(run.resolve());path.unlink()
  raise
 assert footprint(run)<=CAP
def main():
 started=time.time();access=deny_test();run=OUT/'conteos_cineticos_prefijo_Train_s0';run.mkdir(exist_ok=True)
 assert not (run/'resultado.json').exists(),'Completebuilder must not be rerun'
 assert read(OUT/'prueba_operador_cinetico_GPU_s0/resultado.json')['passed']
 assert read(OUT/'verificacion_operador_cinetico_CPU_s0/resultado.json')['passed']
 source_names=['construir_conteos_cineticos_Train.py','conteos_red_cinetica.py','estados_cineticos_15s.py','operador_red_cinetica.py','operador_red_cinetica_GPU.py']
 sources={name:sha(OUT/name) for name in source_names}
 config=dict(fit_n=951718,batchparents=BATCH,cap_MiB=350,cutoff=1396531073,source_sha256='fc73c4e73788c9b2005bfb48bc18cb86adc64d9aeb52c1f76fd5454523e478ed',sources=sources,heldsourceparents_used=0,Validation=False,Test=False)
 if (run/'config.json').exists():assert read(run/'config.json')==config
 else:
  write(run/'config.json',config)
  for name in sources:(run/name).write_bytes((OUT/name).read_bytes())
 bank=OUT/'memoria_puntos_Train_s0/puntos_Train.npz';assert sha(bank)==config['source_sha256']
 with np.load(OUT/'indices_escala_Train_s0/metadatos_Train.npz') as z:
  fit=z['train_index'][:int(z['fit_n'])];rows=z['row'][:len(fit)];timestamp=z['timestamp'][:len(fit)]
 assert len(fit)==951718 and len(np.unique(fit))==len(fit)
 with np.load(bank) as z:
  offsets=z['offsets'];n=z['parent_gps_n'][fit].astype(np.int64);T=z['time_s'][fit];allrows=z['row'];alltimestamp=z['timestamp']
  assert np.array_equal(rows,allrows[fit]) and np.array_equal(timestamp,alltimestamp[fit])
  assert np.isin(fit,z['eligible']).all() and np.array_equal((n-1)*15,T) and np.all(timestamp+T<1396531073)
  original=z['xy_m']
 expected_intervals=int(np.sum(n-1));assert expected_intervals==38725335
 # Independently require newbatchencoder to reproduce all6 alreadyverifiedsamplecountarrays exactly.
 sample=OUT/'verificacion_estados_cineticos_Train_s0'
 with np.load(sample/'estados_ajuste4096.npz') as z:
  K=int(read(sample/'resultado.json')['theoretical_kinetic_states'])
  check=encode_batch(z['kinetic_state'],z['valid_auxiliary_pair'],z['error_m'],z['offsets'],z['timestamp'],K)
 with np.load(OUT/'conteos_cineticos_muestra_Train_s0/conteos_fuentes4096.npz') as z:
  for key,value in check.items():assert np.array_equal(value,z[key]),key
 del check
 sampler=PublicRoadSampler();assert 3*sampler.state_offsets[-1]==K
 journal=read(run/'shards.json') if (run/'shards.json').exists() else []
 completed=0
 for item in journal:
  assert item['begin']==completed and sha(run/item['file'])==item['sha256'];completed=item['end']
 print(json.dumps(dict(stage='fullsource_ready',completed=completed,fitparents=len(fit),original_intervals=expected_intervals,seconds=time.time()-started)),flush=True)
 for begin in range(completed,len(fit),BATCH):
  end=min(begin+BATCH,len(fit));parents=fit[begin:end]
  points=np.concatenate([original[offsets[p]:offsets[p+1]] for p in parents]).astype(np.float64)
  local_offsets=np.r_[0,np.cumsum(n[begin:end])]
  assert local_offsets[-1]==len(points) and np.isfinite(points).all()
  matched,_=sampler.match_fit(points,local_offsets);edge,fraction,error,resets=matched
  assert len(edge)==len(points) and (edge>=0).all() and np.isfinite(error).all()
  spatial=sampler.spatial_states(edge,fraction)
  continues=np.ones(len(points)-1,bool);continues[local_offsets[1:-1]-1]=False
  displacement=np.sqrt(np.square(np.diff(points,axis=0)).sum(1))
  distance=fitted_pair_distances(edge,fraction,continues,sampler.raw['u'],sampler.raw['v'],sampler.raw['length'],sampler.D,*sampler.connections)
  valid=continues&(displacement<=500)&(error[:-1]<=80)&(error[1:]<=80)&np.isfinite(distance)&(distance<=750)
  moving=np.zeros(len(points),np.uint8)
  moving[1:]=np.where(displacement<=5,0,np.where(displacement<=150,1,2)).astype(np.uint8)
  moving[local_offsets[:-1]]=moving[local_offsets[:-1]+1]
  state=spatial*3+moving
  counts=encode_batch(state,valid,error,local_offsets,timestamp[begin:end],K)
  assert int(counts['kinetic_pair_counts'].sum(dtype=np.uint64))==int(valid.sum())
  name='shard_'+str(begin).zfill(7)+'_'+str(end).zfill(7)+'.npz'
  path=run/name;bounded_save(path,run,**counts)
  item=dict(file=name,begin=begin,end=end,sourcepoints=len(points),original_intervals=int(continues.sum()),
   valid_intervals=int(valid.sum()),start_observations=int(counts['start_counts'].sum(dtype=np.uint64)),stop_observations=int(counts['stop_counts'].sum(dtype=np.uint64)),
   source_trips_with_reset=int(np.sum(resets>0)),sha256=sha(path),bytes=path.stat().st_size)
  journal.append(item);write(run/'shards.json',journal)
  print(json.dumps(dict(stage='sourcecount_shard',end=end,total=len(fit),valid_intervals=item['valid_intervals'],persisted_MiB=footprint(run)/1024**2,seconds=time.time()-started)),flush=True)
  del points,matched,edge,fraction,error,resets,spatial,continues,displacement,distance,valid,moving,state,counts
  gc.collect()
 del original,sampler;gc.collect()
 assert sum(item['original_intervals'] for item in journal)==expected_intervals
 combined={}
 for name in ['kinetic_pair','stop','start']:
  allkeys=[];allcounts=[]
  for item in journal:
   with np.load(run/item['file']) as z:
    allkeys.append(z[name+'_keys']);allcounts.append(z[name+'_counts'])
  combined[name+'_keys'],combined[name+'_counts']=aggregate_keys(np.concatenate(allkeys),np.concatenate(allcounts))
  del allkeys,allcounts;gc.collect()
 assert int(combined['kinetic_pair_counts'].sum(dtype=np.uint64))==sum(item['valid_intervals'] for item in journal)
 for name in ['start','stop']:
  assert int(combined[name+'_counts'].sum(dtype=np.uint64))==sum(item[name+'_observations'] for item in journal)
 final=run/'conteos_fuentes.npz'
 bounded_save(final,run,**combined,kinetic_state_count=np.array(K,np.int64),
  source_parent=fit,source_row=rows,source_timestamp=timestamp,source_trueT_s=T)
 # Reload everymergedintegerarray before retiring verifiedtemporaryshards.
 with np.load(final) as z:
  for key,value in combined.items():assert np.array_equal(z[key],value)
 finalhash=sha(final);peak=footprint(run)
 retirement=dict(reason='Exactmergedcounts reloaded and conserved; temporarycountshards only, no results or originaldata.',merged_sha256=finalhash,files=journal,bytes=sum(item['bytes'] for item in journal))
 write(run/'retiro_shards_verificados.json',retirement)
 for item in journal:
  path=run/item['file'];assert path.resolve().is_relative_to(run.resolve()) and sha(path)==item['sha256'];path.unlink()
 report=dict(passed=True,scope='Completed951718precutoffFITsourcecounts; no predictivegraphMAE.',
  fit_sourceparents=len(fit),source_points=sum(item['sourcepoints'] for item in journal),original_intervals=expected_intervals,
  valid_auxiliary_intervals=int(combined['kinetic_pair_counts'].sum(dtype=np.uint64)),
  unique_kinetic_contextpairs=len(combined['kinetic_pair_keys']),public_kinetic_states=K,
  start_observations=int(combined['start_counts'].sum(dtype=np.uint64)),stop_observations=int(combined['stop_counts'].sum(dtype=np.uint64)),
  source_trips_with_HMMreset=sum(item['source_trips_with_reset'] for item in journal),
  every_pair_keeps_original15second_clock=True,no_gap_compression=True,sourcecount_encoder_sample_exact=True,
  all_sources_arrive_before1396531073=True,held_sourceparents=0,original_queries_andT_unchanged=True,
  no_population_GPS_or_roadlabel_bank_written=True,artifact_sha256=finalhash,artifact_MiB=final.stat().st_size/1024**2,
  peak_persistent_MiB=peak/1024**2,after_shard_retirement_MiB=footprint(run)/1024**2,hardcap_MiB=350,
  actual_data_files_accessed=sorted(access),queryheld_MAE_computed=False,Validation=False,Test=False,
  seconds=time.time()-started,sources=sources)
 write(run/'resultado.json',report);print((run/'resultado.json').read_text(encoding='utf-8'),flush=True)
if __name__=='__main__':main()
