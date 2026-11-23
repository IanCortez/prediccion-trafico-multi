"""Geometry-only normalization update on the exact same32768Trainqueries; alltime/context stats frozen."""
from base_replanteo import *
import torch,shutil
from tokens_subtrayectos import query_parts_sub
from tokens_subtrayectos_arco import SubtripArcLengthBank
access=deny_test();start=time.time();torch.set_num_threads(2)
proof=read(OUT/'verificacion_tokens_arco_Train.json');assert proof['passed'] and proof['token_sha256']==sha(OUT/'tokens_subtrayectos_arco.py')
folder=OUT/'constructor_tokens_arco_s0';assert not folder.exists();folder.mkdir()
for name in ['normalizar_tokens_arco_Train.py','tokens_subtrayectos_arco.py']:shutil.copy2(OUT/name,folder/name)
oldfile=OUT/'indices_subtrayectos_s0/constructor_tokens.npz';old=dict(np.load(oldfile));ii=old['normalization_query_indices']
assert len(ii)==32768
data=OUT/'indices_subtrayectos_s0';meta=read(data/'resultado.json');assert sha(data/'train_indices.npz')==meta['outputs']['train']['sha256']
idx=dict(np.load(data/'train_indices.npz'));d=dict(np.load(DATA/'train_x.npz'))
for key in ['row','timestamp']:assert np.array_equal(d[key],idx[key])
part=query_parts_sub(d);bank=SubtripArcLengthBank('cpu')
sums=np.zeros(16);squares=sums.copy();count=0
for first in range(0,len(ii),512):
 take=ii[first:first+512]
 raw=bank.features(*[torch.as_tensor(part[key][take]) for key in ['xy_m','hour','weekend']],
  idx['parent'][take],idx['first'][take],idx['last'][take]).float().numpy()
 valid=raw[:,:,27]>.5;v=raw[:,:,:16][valid].astype(float);assert np.isfinite(v).all()
 sums+=v.sum(0);squares+=(v*v).sum(0);count+=len(v)
 if first%8192==0:print(json.dumps(dict(Trainqueries=first+len(take),seconds=time.time()-start)),flush=True)
mean=old['mean'].copy();std=old['std'].copy();mean[:16]=sums/count
std[:16]=np.maximum(np.sqrt(np.maximum(squares/count-(sums/count)**2,0)),.03)
target=folder/'constructor_tokens.npz'
np.savez_compressed(target,mean=mean,std=std,time_anchor=old['time_anchor'],normalization_query_indices=ii)
assert np.array_equal(mean[16:],old['mean'][16:]) and np.array_equal(std[16:],old['std'][16:])
result=dict(passed=True,normalization_n=32768,valid_source_tokens=count,geometry_columns_updated=list(range(16)),
 all_time_context_statistics_and_anchor_preserved_exactly=True,time_anchor=old['time_anchor'],
 original_constructor_sha256=sha(oldfile),constructor_sha256=sha(target),
 sources={name:sha(OUT/name) for name in ['tokens_subtrayectos.py','tokens_subtrayectos_arco.py','normalizar_tokens_arco_Train.py']},
 semantic_proof_sha256=sha(OUT/'verificacion_tokens_arco_Train.json'),source_indices_manifest_sha256=sha(data/'resultado.json'),
 no_Validation_or_Test=True,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(folder/'resultado.json',result);print(json.dumps(read(folder/'resultado.json')),flush=True)
