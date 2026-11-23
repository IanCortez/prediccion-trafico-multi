"""Full Train geometry teacher for a genuinely global (non-autoregressive) alternative."""
from base_replanteo import *
import shutil
from numba import set_num_threads
from consulta_mapa2014 import ORIGIN
from firma_global90 import geometry90
access=deny_test();start=time.time();set_num_threads(2)
folder=OUT/'datos_firma_global90_Train_s0';assert not folder.exists();folder.mkdir();(folder/'fuentes').mkdir()
for name in ['preparar_firma_global90_Train.py','firma_global90.py']:shutil.copy2(OUT/name,folder/'fuentes'/name)
core=OUT/'datos_calles_parciales_Train_s0';cm=read(core/'resultado.json')
d=dict(np.load(DATA/'train_x.npz'));y=np.load(DATA/'train_y.npy');v=np.load(DATA/'validation_x.npz')
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index'];assert len(eligible)==1023485
assert np.max(d['timestamp'][eligible]+y[eligible])<v['timestamp'].min()
c=dict(np.load(DATA/'grafo_compacto_2014.npz'));xy=np.load(DATA/'mapa_dirigido_2014.npz')['xy']
vec=xy[c['junction_raw'][c['chain_dst']]]-xy[c['junction_raw'][c['chain_src']]]
angle=(np.arctan2(vec[:,1],vec[:,0])+2*np.pi)%(2*np.pi);ori=np.floor(angle*8/(2*np.pi)).astype(np.int32)%8
speed=c['chain_length']/np.maximum(c['chain_free'],.001);sb=np.digitize(speed,[4,7,10,14,20]).astype(np.int32)
def query_xy(tag):
 return np.column_stack([(d[tag+'_lon']-ORIGIN[0])*111195*np.cos(np.deg2rad(ORIGIN[1])),(d[tag+'_lat']-ORIGIN[1])*111195])
chord=np.linalg.norm(query_xy('d')-query_xy('o'),axis=1)
n=len(y);features=np.zeros((n,90),np.float32);confidence=np.zeros(n,np.float32);available=np.zeros(n,bool);seen=np.zeros(n,bool)
for shard,item in enumerate(cm['shards']):
 path=core/item['file'];assert sha(path)==item['sha256'];z=dict(np.load(path));ix=z['train_index']
 assert not seen[ix].any() and np.array_equal(z['row'],d['row'][ix]) and np.array_equal(z['true_duration_s'],y[ix])
 value=geometry90(z['offsets'],z['chain'],z['distance_m'],chord[ix],c['chain_length'],c['chain_free'],ori,sb)
 assert value.shape==(len(ix),90) and np.isfinite(value).all() and value.min()>=0
 features[ix]=value;available[ix]=value[:,0]>0;seen[ix]=True
 for j,(a,b) in enumerate(zip(z['offsets'][:-1],z['offsets'][1:])):
  known=z['chain'][a:b]!=65535
  confidence[ix[j]]=float(z['duration_s'][a:b][known].astype(np.float64).sum())/float(y[ix[j]])
 if shard%40==0:print(json.dumps(dict(stage='geometry_only_Train',trips=int(seen.sum()),seconds=time.time()-start)),flush=True)
assert seen.sum()==894105 and not np.any(seen&~np.isin(np.arange(n),eligible))
assert np.all((confidence>=0)&(confidence<=1.00001));confidence=confidence.clip(0,1);confidence[~available]=0
# Existing 50k oracle teacher provides an independently produced cross-check.
old=np.load(OUT/'oraculo_tiempo_calle_train_s0/descriptores_ruta_Train.npz')
aux=np.load(OUT/'oraculo_tiempo_calle_train_s0/auxiliares_Train.npz');ii=aux['train_index']
assert np.array_equal(old['row'],d['row'][ii]) and len(ii)==50000
old_features=old['features'];delta=np.abs(features[ii].astype(np.float64)-old_features)
scale=np.maximum(np.abs(old_features),1.)
max_abs=float(delta.max());max_scaled=float((delta/scale).max())
assert np.allclose(features[ii],old_features,rtol=5e-6,atol=2e-6),(max_abs,max_scaled)
file=folder/'firma90_Train.npz'
np.savez_compressed(file,row=d['row'],timestamp=d['timestamp'],geometry=features,available=available,teacher_confidence=confidence)
assert file.stat().st_size<240*1024**2
result=dict(complete=True,scope='PRIVATE geometry supervision for Train, never query inputs.',
 original_primary_Train=1023485,Train_rows=n,aux_source_parents=int(seen.sum()),geometry_available=int(available.sum()),
 geometry_dimensions=90,dtype='float32; noPCA/codebook compression',geometry_uses_only='TrainroadIDs/physicaldistances andpublicmapattributes; geometricroadcount, notGPSsamplecount.',
 original_totalT_unchanged=True,all_primary_queries_retained=True,missing_geometry_masks_aux_only=True,
 teacher_confidence='Knownsource-timefraction weights onlyprivategeometrysupervision; neverapublicfeature orprimaryTweight.',
 old50000_teacher_crosscheck=dict(max_absolute_error=max_abs,max_relative_scaled_error=max_scaled,allclose=True),
 source_core_sha256=sha(core/'resultado.json'),outputs_sha256=sha(file),artifact_MiB=file.stat().st_size/1024**2,
 sources={name:sha(folder/'fuentes'/name) for name in ['preparar_firma_global90_Train.py','firma_global90.py']},
 no_Validation_labels_or_paths_or_Test=True,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(folder/'resultado.json',result);print(json.dumps(result),flush=True)
