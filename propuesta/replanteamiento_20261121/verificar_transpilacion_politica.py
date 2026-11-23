"""Verify optimized public-map candidate features against the exact original training statements."""
from base_replanteo import *
import textwrap
from numba import set_num_threads
from catboost import CatBoostClassifier
from consulta_mapa2014 import resources
from rutas_politica_vial import candidate_features
access=deny_test();start=time.time();set_num_threads(2)
path=OUT/'piloto_politica_vial_train_s0/fuente_ejecutada.py';assert sha(path)==read(path.parent/'resultado.json')['source_sha256']
body=path.read_text(encoding='utf-8-sig');a=body.index('  cand=np.asarray(candidates,np.int32)');b=body.index('  features.append(values)',a);code=compile(textwrap.dedent(body[a:b]),str(path),'exec')
raw,c,sample,tree,routes,profiles=resources();set_num_threads(2);D,conn,routing=routes['length'];g=dict(np.load(OUT/'banco_valor_vial_s0/grafo_estatico.npz'))
static=g['static'];nroad=len(static);length=c['chain_length'];sx=raw['xy'][c['junction_raw'][c['chain_src']]];dx=raw['xy'][c['junction_raw'][c['chain_dst']]];v=dx-sx;heading=v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),.001)
rng=np.random.default_rng(20261122);original=[];optimized=[]
for j in range(1024):
 cur=int(rng.integers(nroad));origin=int(rng.integers(nroad));destination=cur if j%4==0 else int(rng.integers(nroad))
 candidates=g['neighbors'][cur][g['neighbor_valid'][cur]].tolist()
 if cur==destination:candidates.append(nroad)
 if not candidates:continue
 timestamp=int(rng.integers(1370000000,1390000000));d=dict(timestamp=np.array([timestamp]));tridx=np.array([0]);i=0
 context=dict(cur=cur,origin=origin,destination=destination,candidates=candidates,nroad=nroad,length=length,D=D,c=c,static=static,heading=heading,sx=sx,dx=dx,np=np,d=d,tridx=tridx,i=i)
 exec(code,context);expected=context['values'];departure=timestamp%86400/86400*2*np.pi
 actual,cost=candidate_features(cur,destination,origin,np.asarray(candidates,np.int32),static,sx,dx,heading,length,D,c['chain_dst'],c['chain_src'],departure)
 original.append(expected);optimized.append(actual)
original=np.concatenate(original);optimized=np.concatenate(optimized);error=float(np.max(np.abs(original-optimized)));assert error<1e-5
model=CatBoostClassifier();model.load_model(str(OUT/'piloto_politica_vial_train_s0/politica.cbm'))
a=model.predict(original,prediction_type='RawFormulaVal');b=model.predict(optimized,prediction_type='RawFormulaVal');score_error=float(np.max(np.abs(a-b)));assert score_error<1e-8
r=dict(passed=True,random_public_states=1024,candidate_rows=len(original),original_training_statements_executed_exactly=True,
 max_candidate_feature_error=error,max_native_model_score_error=score_error,source_sha256=sha(path),generator_sha256=sha(OUT/'rutas_politica_vial.py'),no_real_trajectories_or_labels_decoded=True,
 Validation_access=False,Test_access=False,seconds=time.time()-start)
write(OUT/'verificacion_transpilacion_politica.json',r);print(json.dumps(r),flush=True)
