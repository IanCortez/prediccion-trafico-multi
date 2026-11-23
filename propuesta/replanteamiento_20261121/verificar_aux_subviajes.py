"""Train-only CPU gate for the two tasks and their information/gradient boundaries."""
from base_replanteo import *
import torch
from inferencia import feature_input
from modelo_mezcla_rutas import make_duration_model
from modelo_aux_subviajes import SubtripAuxiliary,duration_score
access=deny_test();start=time.time();torch.set_num_threads(2);torch.manual_seed(20261122)
data=OUT/'subviajes_train_s0';record=read(data/'resultado.json')
assert record['exact_interval_labels'] and record['all_parent_GPS_excluded_from_own_history_bank']
D=np.load(data/'features_train.npz');A=D['X'];p=dict(np.load(data/'consultas_etiquetas_procedencia.npz'));y=p['y']
indices=[record['features'].index(n) for n in ['mm_length_gps_time_600s','mm_free_gps_time_600s','projected_free_freeflow_600s','od_km']]
anchors=[]
for col,scale in zip(indices,[600,600,600,1000]):
 logx=np.log(np.maximum(A[:,col].astype(float)*scale,.1));beta,intercept=np.polyfit(logx,np.log(y),1)
 intercept+=np.log(np.median(y/np.exp(intercept+beta*logx)));anchors.append([intercept,beta])
ctor=dict(anchors=anchors,anchor_indices=indices,k=16,width=512)
torch.save(ctor,data/'constructor_aux.pt')
rec=read(BEST/'resultado.json');ck=torch.load(BEST/'modelo.pt',map_location='cpu',weights_only=False)
duration=make_duration_model(ck,rec['config']);duration.load_state_dict(ck['state']);duration.eval()
model=SubtripAuxiliary(duration,**ctor).eval()
tr,X,names=feature_input('train',rec);tx=torch.from_numpy(X[:3]);ax=torch.from_numpy(A[:3]);t=torch.from_numpy(y[:3].astype(np.float32))
with torch.no_grad():
 before=duration(tx)[0];after=model(tx)[0]
assert torch.equal(before,after)
ap=model.auxiliary(ax);assert ap.shape==(3,16,116) and torch.isfinite(ap).all()
normalization=float((ap.exp().sum(-1)-1).abs().max());assert normalization<1e-5
loss=duration_score(ap,t,model.duration.duration_grid);loss.backward()
shared=sum(float(z.grad.norm()) for n,z in duration.named_parameters() if not n.startswith('backbone.output') and z.grad is not None)
assert shared>0 and model.aux_weight.grad.norm()>0
assert all(z.grad is None for z in duration.backbone.output.parameters())
assert all(torch.isfinite(z.grad).all() for z in model.parameters() if z.grad is not None)
model.zero_grad(set_to_none=True)
for z in duration.parameters():z.requires_grad_(False)
loss=duration_score(model.auxiliary(ax),t,model.duration.duration_grid);loss.backward()
assert all(z.grad is None for z in duration.parameters()) and model.aux_weight.grad.norm()>0
with torch.no_grad():assert torch.equal(model(tx)[0],before)
r=dict(passed=True,scope='Train-only CPU checks, no validation performance',primary_output_exactly_unchanged_by_wrapper=True,
 auxiliary_head_separate=True,auxiliary_gradient_reaches_shared_encoder=True,auxiliary_gradient_does_not_reach_primary_output_head=True,
 warmup_frozen_backbone_has_no_gradients=True,all_gradients_finite=True,probability_normalization_max_error=normalization,
 aux_examples=len(y),Train_only_anchors=True,query=['o_lon','o_lat','d_lon','d_lat','timestamp'],
 model_source_sha256=sha(OUT/'modelo_aux_subviajes.py'),constructor_sha256=sha(data/'constructor_aux.pt'),
 Validation_access=False,Test_access=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(OUT/'verificacion_aux_subviajes.json',r);print(json.dumps(r),flush=True)
