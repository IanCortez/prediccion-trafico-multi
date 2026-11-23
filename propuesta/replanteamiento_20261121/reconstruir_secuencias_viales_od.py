"""Rebuild discarded public-road caches from retained unchanged OD query archives."""
from base_replanteo import *
from secuencias_viales_od import build_public_sequences
access=deny_test();run=OUT/'secuencias_viales_od_s0';record=read(run/'resultado.json')
assert sha(OUT/'secuencias_viales_od.py')==record['source_sha256']['secuencias_viales_od.py']
keys=['o_lon','o_lat','d_lon','d_lat','timestamp']
for split in ['train','validation']:
 path=run/(split+'_secuencias.npz')
 if path.exists():assert sha(path)==record['outputs'][path.name]['sha256'];continue
 d=dict(np.load(DATA/(split+'_x.npz')));n=len(d['row']);T=np.empty((n,2,32),np.int32);S=np.empty((n,2,4),np.float32)
 for first in range(0,n,100000):
  last=min(first+100000,n)
  items=[{k:int(d[k][i]) if k=='timestamp' else float(d[k][i]) for k in keys} for i in range(first,last)]
  T[first:last],S[first:last]=build_public_sequences(items)
 np.savez_compressed(path,row=d['row'],timestamp=d['timestamp'],tokens=T,summary=S)
 assert sha(path)==record['outputs'][path.name]['sha256']
 print(json.dumps(dict(rebuilt=path.name,bytes=path.stat().st_size,hash_exact=True,Test_access=False)),flush=True)
 del T,S
