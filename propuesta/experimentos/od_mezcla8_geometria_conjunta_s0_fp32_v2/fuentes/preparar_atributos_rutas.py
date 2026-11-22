from common import *
from atributos_ruteados import attribute_features,FIELDS
from consulta_mapa2014 import resources
from numba import set_num_threads
import argparse
p=argparse.ArgumentParser();p.add_argument('--verify-only',action='store_true');a=p.parse_args();started=time.time();set_num_threads(4);raw,comp,_,_,routes,_=resources();attrs=dict(np.load(DATA/'atributos_mapa2014.npz'));q=dict(np.load(DATA/'consultas_proyectadas_2014.npz'));ix=np.random.default_rng(1729).choice(len(q['row']),2000,replace=False) if a.verify_only else np.arange(len(q['row']));parts=[];names=[];checks={}
for kind in ('length','free'):
    D,conn,routing=routes[kind];X,error=attribute_features(q['o_edge'][ix],q['o_fraction'][ix],q['o_error'][ix],q['d_edge'][ix],q['d_fraction'][ix],q['d_error'][ix],raw['u'],raw['v'],raw['length'],raw['seconds'],raw['seconds'] if kind=='free' else raw['length'],D,conn,routing,attrs['road_class'],attrs['lanes'],attrs['flags'],attrs['node_control'],attrs['degree'],kind=='free');assert np.isfinite(X).all();assert error.max()<(.001 if kind=='free' else .1);available=X[:,27]>.5;assert np.max(np.abs(X[available,12:27].sum(1)-1))<1e-5;assert np.all((X[:,5:9]>=0)&(X[:,5:9]<=1.00001));parts.append(X);names+=['attrs_'+kind+'_'+n for n in FIELDS];checks[kind]=dict(max_planning_error=float(error.max()),available_fraction=float(available.mean()),mean_intersections=float(X[:,0].mean()),mean_signals=float(X[:,1].mean()))
report=dict(n=len(ix),verify_only=a.verify_only,checks=checks,attribute_source_sha256=sha(DATA/'atributos_mapa2014_manifiesto.json'),query_cache_sha256=sha(DATA/'consultas_proyectadas_2014.npz'),features=len(names),test_labels_read=False,seconds=time.time()-started)
if a.verify_only:write_json(BASE/'instrumentos/atributos_rutas_contrato.json',report)
else:
    allX=np.column_stack(parts);counts=[len(np.load(DATA/(s+'_x.npz'))['row']) for s in ('train','validation','test')];bounds=np.r_[0,np.cumsum(counts)];hashes={}
    for i,split in enumerate(('train','validation','test')):
        lo,hi=bounds[i:i+2];path=DATA/(split+'_atributos_rutas2014.npz');assert not path.exists();np.savez_compressed(path,row=q['row'][lo:hi],X=allX[lo:hi],names=np.array(names));hashes[path.name]=sha(path)
    report.update(outputs=hashes,code_hashes={name:sha(BASE/name) for name in ('preparar_atributos_rutas.py','atributos_ruteados.py')});write_json(DATA/'atributos_rutas2014_manifiesto.json',report)
print(json.dumps(report),flush=True)
