"""Fit raw-token normalization on a fixed eligibleTrain sample, no heldout labels."""
from base_replanteo import *
import torch,shutil
from historia_individual import IndividualHistoryBank,TOKEN_NAMES
access=deny_test();start=time.time();torch.set_num_threads(2);run=OUT/'datos_historia_individual_s0';meta=read(run/'resultado.json')
assert meta['all_Train_sources_exclude_query_UTCweekfold']
file=run/'train_indices.npz';assert sha(file)==meta['outputs'][file.name]['sha256'];z=dict(np.load(file))
memory=Path(meta['shared_memory_file']);bank=IndividualHistoryBank(memory,meta['memory_sha256'])
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index']
sample=np.sort(np.random.default_rng(20261122).choice(eligible,32768,replace=False))
sum_=np.zeros(28,np.float64);sq=np.zeros(28,np.float64);count=0
for first in range(0,len(sample),256):
 ii=sample[first:first+256];ids=z['indices'][ii];safe=np.maximum(ids,0);fold=(z['timestamp'][ii]//604800)%5
 assert ((bank.memory['fold'][safe]!=fold[:,None])|(ids<0)).all()
 R=bank.features(torch.tensor(z['xy_km'][ii]),torch.tensor(z['hour'][ii]),torch.tensor(z['weekend'][ii]),torch.tensor(ids)).float().numpy()
 valid=R[:,:,27]>.5;values=R[valid].astype(np.float64);assert np.isfinite(values).all()
 sum_+=values.sum(0);sq+=np.square(values).sum(0);count+=len(values)
mean=sum_/count;std=np.sqrt(np.maximum(sq/count-mean*mean,.0001));assert np.isfinite(std).all()
oldctor=dict(np.load(OUT/'datos_prototipos_historicos_s0/constructor_Train_fp64.npz'))
dest=run/'normalizacion_Train.npz';assert not dest.exists()
np.savez_compressed(dest,mean=mean.astype(np.float32),std=std.astype(np.float32),time_anchor=oldctor['time_anchor'])
shutil.copy2(OUT/'historia_individual.py',run/'fuente_constructor_tokens.py');shutil.copy2(__file__,run/'fuente_normalizacion.py')
result=dict(query_sample_n=32768,individual_source_tokens=count,all_source_folds_differ_from_query=True,
 fit='EligibleTrain only; source historical labels allowed; no query labels or geometry.',feature_names=TOKEN_NAMES,
 canonical_tokens='Float16 after deterministicFloat32 elementwise feature construction; neural calculationsFloat32.',
 normalization_sha256=sha(dest),token_constructor_sha256=sha(run/'fuente_constructor_tokens.py'),code_sha256=sha(run/'fuente_normalizacion.py'),
 memory_sha256=meta['memory_sha256'],sample_row_sha256=hashlib.sha256(z['row'][sample].tobytes()).hexdigest(),
 Validation_access=False,Test_access=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start,
 next_required='Implement and numerically verify frozen-parent8slot attention; this preprocessing alone is not predictive evidence.')
write(run/'normalizacion_Train_resultado.json',result)
print(json.dumps(dict(query_sample_n=32768,individual_source_tokens=count,normalization_finite=True,seconds=result['seconds'])),flush=True)
