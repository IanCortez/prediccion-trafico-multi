"""Fit-only route geometry residual codes; public signatures never consume actualqueryroute."""
from base_replanteo import *
import shutil
from numba import set_num_threads
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from consulta_mapa2014 import resources,ORIGIN
from rutas_politica_vial import generate_queries
access=deny_test();start=time.time();set_num_threads(2);threadpool_limits(2);run=OUT/'codigos_ruta_global_train_s0';assert not run.exists();run.mkdir();shutil.copy2(__file__,run/'fuente_ejecutada.py')
source=OUT/'oraculo_tiempo_calle_train_s0';oracle=read(source/'resultado.json');assert oracle['passed']
with np.load(source/'auxiliares_Train.npz') as z:q={key:z[key] for key in ['row','train_index','timestamp','true']}
with np.load(source/'descriptores_ruta_Train.npz') as z:
 assert np.array_equal(z['row'],q['row']);actual=z['features']
assert actual.shape==(50000,90)
plan=dict(scope='Train-only latentroute teacher preparation, not predictionMAE.',fit_n=40000,held_n=10000,seed=20261122,
 public='Projectionpriorweightedmean90streetgeometry descriptors over16publicshortest4x4endpointpairs. No learnedpolicy or actualqueryroute in publicsignature.',
 teacher='Actual90routegeometry minus public90signature; StandardScalerfloor.01, fit-onlyPCA12 thenfit-onlyKMeans16. NoTused inPCA/codeconstruction. Unknownpublicsignature getsauxfallbackcode16.',
 latent='Whole-route category supervises p(C|OD) and conditionalTT. Actualheldcode never enters forecast.',
 codebook_fit='Onlyfirst40000chronologicallypurgedfitTrain rows. Held10000actualgeometry used onlytoassigndiagnostic/teacherlabels afterfitfrozen.',
 invariants='Everyquery/trueT retained, raw5publicinputsunchanged, noactualN/timeusedinpublicsignature.',Validation=False,Test=False,
 resources='CPU2, chunks2048, <=100MiBnewartifacts.')
write(run/'plan.json',plan)
raw,c,sample,tree,routes,profiles=resources();set_num_threads(2);D,conn,routing=routes['length'];g=dict(np.load(OUT/'banco_valor_vial_s0/grafo_estatico.npz'))
sx=raw['xy'][c['junction_raw'][c['chain_src']]];dx=raw['xy'][c['junction_raw'][c['chain_dst']]];vec=dx-sx;heading=vec/np.maximum(np.linalg.norm(vec,axis=1,keepdims=True),.001)
angle=(np.arctan2(vec[:,1],vec[:,0])+2*np.pi)%(2*np.pi);orientation=np.floor(angle*8/(2*np.pi)).astype(np.int32)%8
speed=c['chain_length']/np.maximum(c['chain_free'],.001);speedbin=np.digitize(speed,[4,7,10,14,20]).astype(np.int32)
flat=c['flat_incoming_arc'];valid=flat>=0;pc=np.repeat(np.arange(len(c['chain_src']),dtype=np.int32),np.diff(c['chain_offsets']))
edge_chain=np.full(len(raw['u']),-1,np.int32);edge_chain[flat[valid]]=pc[valid]
d=dict(np.load(DATA/'train_x.npz'));ix=q['train_index'];assert np.array_equal(d['row'][ix],q['row']) and np.array_equal(d['timestamp'][ix],q['timestamp'])
with np.load(OUT/'banco_valor_vial_s0/train_proyecciones.npz') as z:bank={key:z[key][ix] for key in ['chain','fraction','error_m','reachable']}
chord=np.linalg.norm(np.column_stack([(d['d_lon'][ix]-d['o_lon'][ix])*111195*np.cos(np.deg2rad(ORIGIN[1])),(d['d_lat'][ix]-d['o_lat'][ix])*111195]),axis=1);departure=np.zeros(len(ix),np.float64)
# max_steps=0 invokes only publicshortesttail; emptytrees prove no learnedroadpolicy can affect construction.
args=(g['static'],sx,dx,heading,c['chain_length'],c['chain_free'],orientation,speedbin,g['neighbors'],g['neighbor_valid'],
 D,conn,routing,raw['u'],raw['v'],raw['length'],c['chain_src'],c['chain_dst'],c['chain_offsets'],c['prefix_length'],c['flat_incoming_arc'],edge_chain,
 np.empty((0,1),np.int32),np.empty((0,1),np.float64),np.empty((0,2),np.float64),1.,0.,0)
chains=bank['chain'].astype(np.int32);fractions=bank['fraction'].astype(np.float32)
public=np.empty((len(ix),90),np.float32);available=np.empty(len(ix),bool)
for a in range(0,len(ix),2048):
 b=min(a+2048,len(ix));value,bad,tail=generate_queries(chains[a:b],fractions[a:b],bank['reachable'][a:b],chord[a:b],departure[a:b],*args)
 eo=bank['error_m'][a:b,0].astype(np.float64);ed=bank['error_m'][a:b,1].astype(np.float64);logits=(-.5*((eo[:,:,None]/30)**2+(ed[:,None,:]/30)**2)).reshape(-1,16)
 logits=np.where(~bad,logits,-1e9);weights=np.exp(logits-logits.max(1,keepdims=True))*(~bad);weights/=np.maximum(weights.sum(1,keepdims=True),1e-20)
 public[a:b]=np.einsum('bk,bkf->bf',weights,value).astype(np.float16).astype(np.float32);available[a:b]=(~bad).any(1)
 if b%8192==0 or b==len(ix):print(json.dumps(dict(stage='public_shortest_signatures',queries=b,total=len(ix),seconds=time.time()-start)),flush=True)
assert np.isfinite(public).all()
nf=40000;residual=actual-public;fitmask=available[:nf];mean=residual[:nf][fitmask].mean(0,dtype=np.float64);std=residual[:nf][fitmask].std(0,dtype=np.float64).clip(.01)
normal=(residual-mean)/std;pca=PCA(n_components=12,svd_solver='full',random_state=20261122);pca.fit(normal[:nf][fitmask]);latent=pca.transform(normal)
km=KMeans(n_clusters=16,n_init=10,max_iter=300,random_state=20261122);km.fit(latent[:nf][fitmask])
codes=np.full(len(ix),16,np.int16);codes[available]=km.predict(latent[available]).astype(np.int16)
assert all(k in codes[:nf] for k in range(16))
constructor=run/'constructor_codigo.npz';np.savez_compressed(constructor,mean=mean,std=std,pca_mean=pca.mean_,pca_components=pca.components_,centers=km.cluster_centers_,unknown_code=16)
# Manual fit-only constructor reproduces labels, including held diagnosticcodes.
manual=((normal-pca.mean_)@pca.components_.T);nearest=((manual[:,:,None]-km.cluster_centers_.T[None])**2).sum(1).argmin(-1)
assert np.array_equal(nearest[available],codes[available])
file=run/'codigos_y_firmas_Train.npz';np.savez_compressed(file,row=q['row'],train_index=ix,timestamp=q['timestamp'],public_signature=public.astype(np.float16),public_available=available,code=codes,fit_n=nf)
result=dict(complete=True,scope=plan['scope'],fit_n=nf,held_n=10000,public_signature_shape=list(public.shape),unknown_fit_n=int((~available[:nf]).sum()),unknown_held_n=int((~available[nf:]).sum()),
 fit_code_counts=np.bincount(codes[:nf],minlength=17),held_code_counts=np.bincount(codes[nf:],minlength=17),PCA_explained_variance=float(pca.explained_variance_ratio_.sum()),
 no_time_labels_in_code_construction=True,all_query_labels_unchanged=True,manual_code_constructor_exact=True,
 source_sha256=sha(run/'fuente_ejecutada.py'),generator_sha256=sha(OUT/'rutas_politica_vial.py'),constructor_sha256=sha(constructor),outputs_sha256=sha(file),
 original_actual_geometry_sha256=sha(source/'descriptores_ruta_Train.npz'),Validation_access=False,Test_access=False,seconds=time.time()-start)
write(run/'resultado.json',result);print(json.dumps({k:result[k] for k in ['complete','fit_n','held_n','unknown_fit_n','unknown_held_n','PCA_explained_variance','seconds']}),flush=True)
