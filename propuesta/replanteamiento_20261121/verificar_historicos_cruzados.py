"""Raw five-field, full-profile reproduction and invariance gate; no labels/GPU."""
from base_replanteo import *
from inferencia import feature_input
from consulta_od import od_records,build_od_features
from consulta_historicos_cruzados import prepare,history_features,MOVEMENT_NAMES,STATIC_NAMES
access=deny_test();start=time.time();rec=read(BEST/'resultado.json')
va,V,names=feature_input('validation',rec);assert names==rec['features']
ii=np.sort(np.random.default_rng(20261122).choice(len(V),65,replace=False))
columns=['o_lon','o_lat','d_lon','d_lat','timestamp']
items=[{k:int(va[k][i]) if k=='timestamp' else float(va[k][i]) for k in columns} for i in ii]
raw=od_records(items);rawX,rnames=build_od_features(raw,rec['config']);assert rnames==names
p=prepare(items);mi=[names.index(n) for n in MOVEMENT_NAMES];si=[names.index(n) for n in STATIC_NAMES]
M5,S5=history_features(p,5)
m_error=float(np.abs(M5-V[ii][:,mi]).max());s_error=float(np.abs(S5-V[ii][:,si]).max())
assert m_error<1e-5 and s_error<1e-5
assert float(np.abs(rawX-V[ii]).max())<1e-5
diversity=[]
for fold in range(5):
 M,S=history_features(p,fold);assert np.array_equal(S,S5)
 diversity.append(float(np.abs(M-M5).max()))
rev=prepare(items[::-1]);mr,sr=history_features(rev,2);m2,s2=history_features(p,2)
assert np.array_equal(mr[::-1],m2) and np.array_equal(sr[::-1],s2)
for index in [0,1,63,64]:
 one=prepare([items[index]]);mo,so=history_features(one,2)
 assert np.array_equal(mo[0],m2[index]) and np.array_equal(so[0],s2[index])
rejected=False
try:prepare([dict(items[0],call='A')])
except AssertionError:rejected=True
assert rejected and np.array_equal(p['d']['timestamp'],np.array([i['timestamp'] for i in items]))
assert max(diversity)>0
result=dict(passed=True,scope='Raw query/input construction gate; no primary MAE or GPU run',
 query_fields=columns,observed_call_rejected=True,departure_timestamp_unmodified=True,
 n=65,full_profile_movement_max_error=m_error,static_max_error=s_error,whole_raw_feature_max_error=float(np.abs(rawX-V[ii]).max()),
 static_features_equal_across_all_history_views=True,history_movement_diversity_vs_full=diversity,
 query_order_and_singleton_invariance=True,Validation_labels_read=False,Test_access=False,
 source_sha256=sha(OUT/'consulta_historicos_cruzados.py'),code_sha256=sha(__file__),actual_files_accessed=sorted(access),seconds=time.time()-start)
write(OUT/'verificacion_historicos_cruzados.json',result);print(json.dumps(result),flush=True)
