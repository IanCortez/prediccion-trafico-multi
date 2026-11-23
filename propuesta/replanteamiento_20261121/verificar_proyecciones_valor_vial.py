"""Train-only raw endpointset bank/singleton/order exactness; no GPU or target decoding."""
from base_replanteo import *
from proyecciones_valor_vial import project_street_query
access=deny_test();start=time.time();d=dict(np.load(DATA/'train_x.npz'));z=dict(np.load(OUT/'banco_valor_vial_s0/train_proyecciones.npz'))
ii=np.sort(np.random.default_rng(20261122).choice(len(d['row']),65,replace=False));fields=['o_lon','o_lat','d_lon','d_lat','timestamp']
q={key:d[key][ii] for key in fields};raw=project_street_query(q);reverse=project_street_query({key:value[::-1] for key,value in q.items()});single=project_street_query({key:value[:1] for key,value in q.items()})
for key,value in raw.items():
 assert np.array_equal(value,z[key][ii]),key
 assert np.array_equal(value,reverse[key][::-1]),key
 assert np.array_equal(value[:1],single[key]),key
result=dict(passed=True,n=65,Train_only=True,raw_public4candidate_chain_fraction_error_distance_reachability_exact=True,singleton_reverse_exact=True,
 source_sha256=sha(OUT/'proyecciones_valor_vial.py'),bank_manifest_sha256=sha(OUT/'banco_valor_vial_s0/resultado.json'),Validation_access=False,Test_access=False,seconds=time.time()-start)
write(OUT/'verificacion_proyecciones_valor_vial.json',result);print(json.dumps(result),flush=True)
