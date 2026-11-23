"""Full pre-cutoff Train source references, only after the smallpilot is closed/replayed."""
from base_replanteo import *
import gc,shutil
from recuperar_fuentes_ordenadas import OrderedSourceMemory
access=deny_test();start=time.time()
oldrun=OUT/'mezcla_individual_mediana_Train_s0'
outcome=read(oldrun/'resultado.json');replay=read(oldrun/'reproduccion_independiente.json')
assert outcome['complete'] and replay['passed'] and not replay['stable_practical_gate_passed']
train_last=read(oldrun/'none/historia.json')[-1]['train_mae_s']
assert replay['stable_metrics']['none']['mae_s']-train_last>20
plan=read(OUT/'plan_escala_datos_Train.json');run=OUT/'indices_escala_Train_s0'
assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
for name in ['preparar_indices_escala_Train.py','recuperar_fuentes_ordenadas.py','recuperar_subtrayectos.py','base_replanteo.py']:
 shutil.copy2(OUT/name,run/'fuentes'/name)
with np.load(OUT/'memoria_puntos_Train_s0/puntos_Train.npz') as z:
 eligible=z['eligible'];stamps=z['timestamp'];source_time=z['time_s']
fit=np.sort(eligible[stamps[eligible]+source_time[eligible]<plan['cutoff']])
with np.load(OUT/'oraculo_tiempo_calle_train_s0/auxiliares_Train.npz') as z:held=z['train_index'][40000:]
assert len(fit)==plan['source_fit_n'] and len(held)==10000 and np.intersect1d(fit,held).size==0
index=np.r_[fit,held];nfit=len(fit);n=len(index);tr=dict(np.load(DATA/'train_x.npz'))
assert (tr['timestamp'][fit]+np.load(DATA/'train_y.npy')[fit]<plan['cutoff']).all()
dd={key:tr[key][index] for key in ['o_lon','o_lat','d_lon','d_lat','timestamp']}
fold=(dd['timestamp']//604800)%5
parent=np.full((n,64),-1,np.int32);first=np.zeros((n,64),np.uint8);last=np.zeros_like(first)
memory=OrderedSourceMemory(source_arrival_before=plan['cutoff']);stats=[]
write(run/'plan.json',dict(source='Full existing eligibleprimaryTrainarriving before originalheldcutoff; originalTrainpilotheld10k unchanged.',
 fit_n=nfit,held_n=10000,cutoff=plan['cutoff'],trigger_verified=True,fold_exclusion_before_tree=True,sourceK=2048,
 cache_max_MiB=350,no_rawtoken_cache=True,all_original_primary_queries_filters_labels_preserved=True,Validation=False,Test=False))
for group in range(6):
 selected=np.flatnonzero((np.arange(n)<nfit)&(fold==group)) if group<5 else np.arange(nfit,n)
 memory.build_tree(group);evaluated=0;discovered=0;close=0;done=0
 for a in range(0,len(selected),256):
  ids=selected[a:a+256];queries={k:v[ids] for k,v in dd.items()}
  result=memory.query(queries,training=group<5,diagnostics=True)
  parent[ids]=result[0];first[ids]=result[1];last[ids]=result[2]
  valid=result[0]>=0;have=result[0][valid]
  assert memory.available[have].all()
  if group<5:assert (memory.data['parent_fold'][have]!=group).all()
  assert not np.any((result[0]==index[ids,None])&valid)
  evaluated+=int(result[7].sum());discovered+=int(result[8].sum());close+=int((result[9]<=100).sum());done+=len(ids)
  if (a//256)%64==0:print(json.dumps(dict(stage='full_Train_source_indices',fold=group,done=done,total=len(selected),seconds=time.time()-start)),flush=True)
 stats.append(dict(fold=group,n=done,mean_evaluated=evaluated/done,mean_discovered=discovered/done,within100m_fraction=close/done))
 write(run/'progress.json',dict(completed_groups=stats,seconds=time.time()-start))
# Heldreferences must stay byteidentical because queries, sourcecutoff and sourcealgorithm are unchanged.
old=OUT/'piloto_cdf_ordenada_Train_s0/indices_fuentes.npz'
with np.load(old) as z:
 assert np.array_equal(z['row'][40000:],tr['row'][held])
 for key,array in [('parent',parent),('first',first),('last',last)]:assert np.array_equal(z[key][40000:],array[nfit:])
# Independent all-reference temporal/week/span/sourceendpoint checks.
valid=parent>=0;safe=np.maximum(parent,0);meta=memory.data
assert (meta['timestamp'][safe[valid]]+meta['time_s'][safe[valid]]<plan['cutoff']).all()
span=last.astype(np.int16)-first.astype(np.int16);assert ((span[valid]>=4)&(span[valid]<=119)).all()
assert np.all(last[valid]<meta['parent_gps_n'][safe[valid]])
origin_point=meta['offsets'][safe]+first;destination_point=meta['offsets'][safe]+last
assert memory.indexable[origin_point[valid]].all() and memory.indexable[destination_point[valid]].all()
del safe,origin_point,destination_point,span,memory;gc.collect()
file=run/'indices_fuentes.npz'
np.savez_compressed(file,row=tr['row'][index],timestamp=dd['timestamp'],train_index=index.astype(np.int32),
 fit_n=np.asarray(nfit,np.int64),parent=parent,first=first,last=last)
assert file.stat().st_size<350*1024**2
result=dict(complete=True,fit_n=nfit,held_n=10000,source_cutoff=plan['cutoff'],statistics=stats,
 held_all10000_refs_identical_to_smallpilot=True,all_source_arrivals_before_cutoff=True,
 wholeparentfold_exclusion_before_tree=True,no_queryself_source=True,sourceendpoint_bbox_and_span_preserved=True,
 original_primary_queries_labels_split_unchanged=True,no_rawtokens_saved=True,
 indices_sha256=sha(file),MiB=file.stat().st_size/1024**2,
 source_pointbank_sha256=read(OUT/'memoria_puntos_Train_s0/resultado.json')['outputs_sha256'],
 sources={p.name:sha(p) for p in (run/'fuentes').glob('*.py')},Validation=False,Test=False,
 actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(run/'resultado.json',result);print(json.dumps(read(run/'resultado.json')),flush=True)
