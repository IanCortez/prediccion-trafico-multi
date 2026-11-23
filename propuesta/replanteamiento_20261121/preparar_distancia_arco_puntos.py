"""Exact metric cumulative distance within each authorized Train parent; no querypath bank."""
from base_replanteo import *
from numba import njit,prange,set_num_threads
import shutil
@njit(cache=True,parallel=True)
def prefix_distance(xy,offsets):
 out=np.zeros(len(xy),np.float32)
 for p in prange(len(offsets)-1):
  first=offsets[p];last=offsets[p+1];total=0.
  for j in range(first+1,last):
   dx=float(xy[j,0])-float(xy[j-1,0]);dy=float(xy[j,1])-float(xy[j-1,1])
   total+=np.sqrt(dx*dx+dy*dy);out[j]=total
 return out
access=deny_test();start=time.time();set_num_threads(2)
folder=OUT/'distancia_arco_puntos_Train_s0';assert not folder.exists();folder.mkdir();shutil.copy2(__file__,folder/'fuente_ejecutada.py')
source=OUT/'memoria_puntos_Train_s0';meta=read(source/'resultado.json');assert sha(source/'puntos_Train.npz')==meta['outputs_sha256']
z=dict(np.load(source/'puntos_Train.npz'));prefix=prefix_distance(z['xy_m'],z['offsets'])
assert len(prefix)==42624621 and np.isfinite(prefix).all() and prefix.min()==0
n=z['parent_gps_n'];have=n>0;starts=z['offsets'][:-1][have];ends=z['offsets'][1:][have]-1
assert np.all(prefix[starts]==0)
ii=np.sort(np.random.default_rng(711).choice(np.flatnonzero(have),1024,replace=False));errors=[]
for p in ii:
 first=z['offsets'][p];last=z['offsets'][p+1]
 direct=np.r_[0,np.cumsum(np.linalg.norm(np.diff(z['xy_m'][first:last].astype(float),axis=0),axis=1))]
 errors.append(np.max(np.abs(prefix[first:last]-direct)))
assert max(errors)<.02,max(errors)
file=folder/'distancia_acumulada_m.npz';np.savez_compressed(file,prefix_m=prefix)
assert file.stat().st_size<180*1024**2
result=dict(complete=True,points=len(prefix),source_point_bank_sha256=meta['outputs_sha256'],units='metres, zero at each parentstart',
 outputs_sha256=sha(file),MiB=file.stat().st_size/1024**2,max_independent1024parent_error_m=max(errors),
 source_sha256=sha(folder/'fuente_ejecutada.py'),purpose='Correct arclength-sampled historicalsource geometry andfullintervalpathlength; noGPSN/T/querypath changes.',
 no_Validation_or_Test=True,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(folder/'resultado.json',result);print(json.dumps(read(folder/'resultado.json')),flush=True)
