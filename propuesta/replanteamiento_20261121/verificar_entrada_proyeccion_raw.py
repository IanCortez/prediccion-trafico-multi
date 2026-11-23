"""Label-blind five-field query reconstruction for the expanded route mixture."""
from base_replanteo import *
from inferencia import feature_input
from consulta_od import od_records,build_od_features
from rutas_proyeccion_incierta import build_uncertain_routes
access=deny_test();start=time.time();rec=read(BEST/'resultado.json')
d,X,names=feature_input('validation',rec);bank=dict(np.load(OUT/'banco_proyeccion_incierta_s0/validation_banco.npz'))
assert np.array_equal(d['row'],bank['row'])
ii=np.sort(np.random.default_rng(20261122).choice(len(X),65,replace=False));keys=['o_lon','o_lat','d_lon','d_lat','timestamp']
items=[{k:int(d[k][i]) if k=='timestamp' else float(d[k][i]) for k in keys} for i in ii]
raw,names2=build_od_features(od_records(items),rec['config']);C,pairs=build_uncertain_routes(items,training=False);C=C.astype(np.float16)
assert names==names2 and np.max(np.abs(raw-X[ii]))<1e-5
assert np.array_equal(C,bank['bank'][ii]) and np.array_equal(pairs,bank['projection_pairs'][ii])
reverse,_,=build_uncertain_routes(items[::-1],training=False);single,_=build_uncertain_routes([items[0]],training=False)
assert np.array_equal(reverse.astype(np.float16)[::-1],C) and np.array_equal(single.astype(np.float16),C[:1])
rejected=[]
for extra in ['actual_path','point_count','duration','call','parent_row']:
 try:build_uncertain_routes([dict(items[0],**{extra:1})],training=False)
 except AssertionError:rejected.append(extra)
assert len(rejected)==5
r=dict(passed=True,n=65,scope='Label-blind input reconstruction, not checkpoint prediction reproduction.',
 raw_query_fields=keys,global216_feature_max_error=float(np.max(np.abs(raw-X[ii]))),
 candidate_FP16_features_exact=True,projection_pairs_exact=True,reverse_and_singleton_exact=True,
 forbidden_extra_fields_rejected=rejected,Validation_labels_accessed=False,Test_access=False,
 sources={s:sha(OUT/s) for s in ['rutas_proyeccion_incierta.py','consulta_historicos_cruzados.py']},
 actual_data_files_accessed=sorted(access),seconds=time.time()-start)
assert not any(Path(p).name=='validation_y.npy' for p in access)
write(OUT/'verificacion_entrada_proyeccion_raw.json',r);print(json.dumps({k:r[k] for k in ['passed','n','global216_feature_max_error','candidate_FP16_features_exact','seconds']}),flush=True)
