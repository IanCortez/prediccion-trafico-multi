"""Rebuild the bounded auxiliary cache from retained five-field queries, no raw GPS."""
from base_replanteo import *
from consulta_subviajes_train import build_training_features
access=deny_test();run=OUT/'subviajes_train_s0';record=read(run/'resultado.json');path=run/'features_train.npz'
if path.exists():
 assert sha(path)==record['outputs'][path.name]['sha256'];print('Verified existing auxiliary feature cache.')
else:
 for name in ['consulta_subviajes_train.py','consulta_historicos_cruzados.py']:
  assert sha(OUT/name)==record['source_sha256'][name]
 q=dict(np.load(run/'consultas_etiquetas_procedencia.npz'));keys=['o_lon','o_lat','d_lon','d_lat','timestamp']
 items=[{k:int(q[k][i]) if k=='timestamp' else float(q[k][i]) for k in keys} for i in range(len(q['y']))]
 X,names=build_training_features(items,read(BEST/'resultado.json')['config'])
 assert names==record['features'] and X.shape==tuple(record['feature_shape']) and np.isfinite(X).all()
 np.savez_compressed(path,X=X,names=np.asarray(names))
 assert sha(path)==record['outputs'][path.name]['sha256']
 print(json.dumps(dict(rebuilt=True,n=len(X),bytes=path.stat().st_size,sha256=sha(path),Test_access=False)),flush=True)
