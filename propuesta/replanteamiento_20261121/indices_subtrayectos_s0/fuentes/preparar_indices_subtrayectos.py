"""Bounded ordered-source indices for all primary queries; crossfit parents before tree."""
from base_replanteo import *
import gc,shutil
from recuperar_subtrayectos import SubtrajectoryMemory
access=deny_test();start=time.time();folder=OUT/'indices_subtrayectos_s0';folder.mkdir(exist_ok=True)
sources=['preparar_indices_subtrayectos.py','recuperar_subtrayectos.py','datos_progreso_recurrente.py','base_replanteo.py']
if not (folder/'fuentes').exists():
 (folder/'fuentes').mkdir()
 for name in sources:shutil.copy2(OUT/name,folder/'fuentes'/name)
else:
 for name in sources:assert sha(OUT/name)==sha(folder/'fuentes'/name)
plan=dict(k=2048,sources=64,Train='Exclude entire parentweekfold before constructing nearestpointtree',
 raw_query=['o_lon','o_lat','d_lon','d_lat','timestamp'],preserve='Only parentint32 andfirst/lastuint8 perquery, reconstructtokens on demand',max_MiB=500,
 sources_sha256={name:sha(folder/'fuentes'/name) for name in sources})
if (folder/'plan.json').exists():assert read(folder/'plan.json')==plan
else:write(folder/'plan.json',plan)
bank=SubtrajectoryMemory();eligible=bank.data['eligible'];fields=plan['raw_query'];manifest={}
for split in ['train','validation']:
 d=dict(np.load(DATA/(split+'_x.npz')))
 if (folder/(split+'_indices.npz')).exists():
  meta=read(folder/(split+'_resultado.json'));assert sha(folder/(split+'_indices.npz'))==meta['sha256'];manifest[split]=meta;continue
 n=len(d['timestamp']);p=np.full((n,64),-1,np.int32);i=np.zeros((n,64),np.uint8);j=i.copy();chunks=[]
 folds=range(5) if split=='train' else [5]
 for fold in folds:
  selected=eligible[((d['timestamp'][eligible]//604800)%5)==fold] if split=='train' else np.arange(n)
  file=folder/(split+'_fold'+str(fold)+'.npz');infofile=file.with_suffix('.json')
  if not file.exists():
   bank.build_tree(fold);pp=[];ii=[];jj=[]
   for first in range(0,len(selected),128):
    take=selected[first:first+128];source=bank.query({key:d[key][take] for key in fields},training=split=='train')
    pp.append(source[0]);ii.append(source[1]);jj.append(source[2])
    if first%32768==0:print(json.dumps(dict(split=split,fold=fold,completed=first+len(take),total=len(selected),seconds=time.time()-start)),flush=True)
   np.savez_compressed(file,index=selected,parent=np.concatenate(pp),first=np.concatenate(ii),last=np.concatenate(jj))
   write(infofile,dict(sha256=sha(file),n=len(selected),fold=fold));del pp,ii,jj
  z=dict(np.load(file));info=read(infofile);assert sha(file)==info['sha256'] and np.array_equal(z['index'],selected)
  parent=z['parent'];valid=parent>=0;safe=np.maximum(parent,0)
  assert np.all(z['last'][valid].astype(int)-z['first'][valid].astype(int)>=4)
  if split=='train':assert np.all(bank.data['parent_fold'][safe[valid]]!=fold)
  else:assert np.max(bank.data['timestamp'][bank.data['eligible']]+bank.data['time_s'][bank.data['eligible']])<d['timestamp'].min()
  p[selected]=z['parent'];i[selected]=z['first'];j[selected]=z['last'];chunks.append(dict(file=str(file),sha256=info['sha256']))
  gc.collect()
 np.savez_compressed(folder/(split+'_indices.npz'),row=d['row'],timestamp=d['timestamp'],parent=p,first=i,last=j)
 path=folder/(split+'_indices.npz');assert path.stat().st_size<500*1024**2
 counts=(p>=0).sum(1);meta=dict(complete=True,sha256=sha(path),n=n,MiB=path.stat().st_size/1024**2,
  counts_quantiles=np.quantile(counts,[0,.1,.25,.5,.75,.9,1]),eligible_n=len(eligible) if split=='train' else n,
  source_fold_checked=True,all_original_queries_preserved=True)
 write(folder/(split+'_resultado.json'),meta);manifest[split]=meta
 merged=dict(np.load(path))
 for chunk in chunks:
  file=Path(chunk['file']);z=dict(np.load(file));take=z['index']
  for key in ['parent','first','last']:assert np.array_equal(z[key],merged[key][take])
  assert file.resolve().is_relative_to(folder.resolve()) and sha(file)==chunk['sha256'];file.unlink()
 write(folder/(split+'_limpieza.json'),dict(duplicate_fold_files_removed=chunks,merged_checked_exact=True))
 del p,i,j,merged;gc.collect()
write(folder/'resultado.json',dict(complete=True,outputs=manifest,bank_sha256=sha(OUT/'memoria_puntos_Train_s0/puntos_Train.npz'),
 sources=plan['sources_sha256'],Test_access=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start))
print(json.dumps(read(folder/'resultado.json')),flush=True)
