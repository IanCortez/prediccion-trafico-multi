"""Train-only runtime, no-history, source-exclusion and gradient checks for raw64attention."""
from base_replanteo import *
import torch
from inferencia_historica_fresca import load_frozen_history
from inferencia import feature_input
access=deny_test();start=time.time();torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
# Load the protected parent before any live history helper import to bind frozen dependencies.
parent,ck,record,oldrun,oldlib=load_frozen_history('historica_fresca_s0',device='cpu');del oldlib
from historia_individual import IndividualHistoryBank
from modelo_atencion_historia import IndividualHistoryAttention
data=OUT/'datos_historia_individual_s0';meta=read(data/'resultado.json');norm=read(data/'normalizacion_Train_resultado.json')
assert sha(OUT/'historia_individual.py')==norm['token_constructor_sha256']
constructor=dict(np.load(data/'normalizacion_Train.npz'));assert sha(data/'normalizacion_Train.npz')==norm['normalization_sha256']
index=dict(np.load(data/'train_indices.npz'));ii=np.flatnonzero((index['timestamp']//604800)%5==0)[:17]
bank=IndividualHistoryBank(Path(meta['shared_memory_file']),meta['memory_sha256'])
R=bank.features(torch.tensor(index['xy_km'][ii]),torch.tensor(index['hour'][ii]),torch.tensor(index['weekend'][ii]),torch.tensor(index['indices'][ii]))
before=R.clone();source=bank.tensors('cpu');own=torch.tensor(bank.memory['fold']==0)
old_time=source['time_s'][own].clone();old_geometry=source['geometry'][own].clone()
source['time_s'][own]+=100000;source['geometry'][own]+=100
after=bank.features(torch.tensor(index['xy_km'][ii]),torch.tensor(index['hour'][ii]),torch.tensor(index['weekend'][ii]),torch.tensor(index['indices'][ii]))
assert torch.equal(before,after);source['time_s'][own]=old_time;source['geometry'][own]=old_geometry
d,X,names=feature_input('train',read(BEST/'resultado.json'));x=torch.tensor(X[ii]);del X
C=np.load(OUT/'banco_proyeccion_incierta_s0/train_banco.npz')['bank'][ii];H=np.load(OUT/'datos_prototipos_historicos_s0/train_historicos.npz')['history'][ii]
c=torch.tensor(C);h=torch.tensor(H)
items=[{k:int(d[k][i]) if k=='timestamp' else float(d[k][i]) for k in ['o_lon','o_lat','d_lon','d_lat','timestamp']} for i in ii]
assert torch.equal(R,bank.raw(items,training=True))
torch.manual_seed(20261122);model=IndividualHistoryAttention(parent,constructor['mean'],constructor['std'],constructor['time_anchor']).eval()
with torch.no_grad():
 out=model(x,c,h,R);base=parent(x,c,h)
 assert out[1].shape[-1]==30 and torch.max(torch.abs(out[0].exp().sum(-1)-1))<2e-5
 perm=torch.randperm(64);reverse=model(x,c,h,R[:,perm]);error=float((out[0]-reverse[0]).abs().max())
 assert error<2e-5
 missing=model(x,c,h,torch.zeros_like(R));assert torch.equal(missing[0],base[0])
 assert torch.isfinite(out[0]).all()
model.train();out=model(x,c,h,R);target=torch.full((len(x),),300.)
label=((target-60)/15).long();loss=-out[0].gather(-1,label[:,None,None].expand(-1,16,1)).mean();loss.backward()
assert all(p.grad is None for p in parent.parameters())
assert all(torch.isfinite(p.grad).all() for p in model.branch_parameters() if p.grad is not None)
head_norm=sum(float(p.grad.abs().sum()) for p in model.head.parameters() if p.grad is not None);gate_norm=sum(float(p.grad.abs().sum()) for p in model.gate.parameters() if p.grad is not None)
assert head_norm>0 and gate_norm>0
assert not any(Path(q).name.startswith('validation') for q in access)
result=dict(passed=True,scope='17Train only; no predictive gain claimed.',parent_frozen_no_grad=True,
 component_count=30,probabilities_normalized=True,raw_cached_tokens_exact=True,ownfold_value_intervention_exact=True,
 nohistory_parent_probabilities_exact=True,source_permutation_max_log_probability_error=error,
 finite_gradients=True,head_gradient_l1=head_norm,gate_gradient_l1=gate_norm,
 model_sha256=sha(OUT/'modelo_atencion_historia.py'),token_constructor_sha256=sha(OUT/'historia_individual.py'),
 normalization_sha256=norm['normalization_sha256'],parent_sha256=record['model_sha256'],
 Validation_access=False,Test_access=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(OUT/'verificacion_atencion_historia_cpu.json',result)
print(json.dumps(dict(passed=True,source_permutation_max_log_probability_error=error,head_gradient_l1=head_norm,gate_gradient_l1=gate_norm,seconds=result['seconds'])),flush=True)
