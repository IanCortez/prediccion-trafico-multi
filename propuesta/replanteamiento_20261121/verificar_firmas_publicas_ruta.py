"""Raw5 publicfeature replay, respecting the frozenhistory inference-date guard."""
from base_replanteo import *
from firmas_publicas_ruta import raw_public_features,public_signature_from_query,public_signature_from_bank
from consulta_od import od_records
from inferencia import feature_input
access=deny_test();start=time.time();fields=['o_lon','o_lat','d_lon','d_lat','timestamp']
def items_for(d,ix):return [{key:int(d[key][i]) if key=='timestamp' else float(d[key][i]) for key in fields} for i in ix]
z=dict(np.load(OUT/'codigos_ruta_global_train_s0/codigos_y_firmas_Train.npz'));d=dict(np.load(DATA/'train_x.npz'))
ii=np.sort(np.random.default_rng(20261122).choice(50000,65,replace=False));ix=z['train_index'][ii];items=items_for(d,ix)
sig,available=public_signature_from_query(od_records(items));assert np.array_equal(sig,z['public_signature'][ii].astype(np.float32)) and np.array_equal(available,z['public_available'][ii])
reverse,ra=public_signature_from_query(od_records(items[::-1]));single,sa=public_signature_from_query(od_records(items[:1]))
assert np.array_equal(sig,reverse[::-1]) and np.array_equal(sig[:1],single)
# Originalrawfeatureadapter legitimatelyrequiresdates afterthe frozenTrainingwindow becauseit canaccesshistoricalprofiles.
# Keepthatguard; useValidationQUERYFIELDSonly, noValidationduration/routeinteriors toverify fullinputconstruction.
v,V,names=feature_input('validation',read(BEST/'resultado.json'));vi=np.sort(np.random.default_rng(20261122).choice(len(v['row']),65,replace=False));vitems=items_for(v,vi)
raw,rnames,va=raw_public_features(vitems);keep=np.asarray([not n.startswith('mm_') for n in names]);assert keep.sum()==206
assert rnames==[n for n,k in zip(names,keep) if k]+['public_street_signature_'+str(i) for i in range(90)]
error=float(np.max(np.abs(raw[:,:206]-V[vi][:,keep])));assert error<1e-5
with np.load(OUT/'banco_valor_vial_s0/validation_proyecciones.npz') as b:
 assert np.array_equal(b['row'][vi],v['row'][vi]) and np.array_equal(b['timestamp'][vi],v['timestamp'][vi])
 cached_bank={key:b[key][vi] for key in ['chain','fraction','error_m','remaining_km','reachable']}
vsig,vavailable=public_signature_from_bank(od_records(vitems),cached_bank)
assert np.array_equal(raw[:,-90:],vsig) and np.array_equal(va,vavailable)
for key in ['route','N','duration','driver','call']:
 try:od_records([dict(vitems[0],**{key:999})])
 except AssertionError:pass
 else:raise AssertionError('Forbiddenqueryfieldaccepted')
assert not any(Path(p).name=='validation_y.npy' for p in access)
r=dict(passed=True,Train_raw_signature_n=65,Validation_raw_public_features_n=65,
 raw206_validation_features_match_cached_to_1e_5=True,maximum_numeric_feature_error=error,
 Train_public90signature_and_availability_exact=True,signature_reverse_singleton_exact=True,
 Validation_public90signature_matches_independentcached_publicprojection_bank=True,
 original_historicaladapter_dateguard_preserved=True,extra_route_N_time_driver_call_rejected=True,
 no_actualGPSpaths_Validationdurationlabels_or_Test_opened=True,
 source_sha256=sha(OUT/'firmas_publicas_ruta.py'),projection_source_sha256=sha(OUT/'proyecciones_valor_vial.py'),
 generator_sha256=sha(OUT/'rutas_politica_vial.py'),code_manifest_sha256=sha(OUT/'codigos_ruta_global_train_s0/resultado.json'),
 actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(OUT/'verificacion_firmas_publicas_ruta.json',r);print(json.dumps({k:r[k] for k in ['passed','maximum_numeric_feature_error','no_actualGPSpaths_Validationdurationlabels_or_Test_opened','seconds']}),flush=True)
