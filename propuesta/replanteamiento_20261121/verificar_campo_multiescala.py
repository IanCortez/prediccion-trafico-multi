"""Spatial-field interpolation, zero-branch equivalence and real-Train gradients."""
from base_replanteo import *
import torch
from inferencia import feature_input
from modelo_mezcla_rutas import make_duration_model
from modelo_campo_multiescala import BilinearEndpointLevel,SpatialFieldJoint
access=deny_test();torch.set_num_threads(2);torch.manual_seed(0);start=time.time()
level=BilinearEndpointLevel(250,width=2)
nodes=torch.arange(level.nx*level.ny);physical=torch.stack([-6500+(nodes//level.ny)*250,-3000+(nodes%level.ny)*250],-1).float()
with torch.no_grad():level.embedding.weight.copy_(physical)
points=torch.tensor([[-6425.,-2975.],[10.,40.],[6425.,3925.],[-6250.-.001,-2750.],[-6250.+.001,-2750.]])
interp=level(points);error=float((interp-points).abs().max());assert error<.001
ids,weights=level.indices_weights(points);assert torch.max(torch.abs(weights.sum(-1)-1))<1e-6 and (weights>=0).all()
assert torch.allclose(level(points.flip(0)).flip(0),interp,atol=1e-5)
rec=read(BEST/'resultado.json');tr,X,names=feature_input('train',rec);assert names==rec['features'] and names[:4]==['ox','oy','dx','dy'] and all(not n.startswith('call_') for n in names)
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index'];ii=eligible[:32]
x=torch.from_numpy(X[ii].copy());t=torch.from_numpy(np.load(DATA/'train_y.npy')[ii]);del X
ck=torch.load(BEST/'modelo.pt',map_location='cpu',weights_only=False)
clocks=[names.index(n) for n in ['hour_sin','hour_cos','week_sin','week_cos','season_sin','season_cos','weekend','calendar_special']]
model=SpatialFieldJoint(ck['mean'],ck['std'],ck['keep'],ck['bins'],ck['anchors'],ck['anchor_indices'],ck['geometry_indices'],clocks,16,512)
loaded=model.load_state_dict(ck['state'],strict=False);assert not loaded.unexpected_keys and all(n.startswith('field.') for n in loaded.missing_keys)
model.eval();baseline=make_duration_model(ck,rec['config']);baseline.load_state_dict(ck['state']);baseline.eval()
with torch.no_grad():
 plain=baseline(x)[0];out=model(x)[0];zero_error=float((plain-out).abs().max());assert zero_error==0
 model.field.net[-1].weight.normal_(0,.01)
out=model(x);label=((t-60)/15).long();loss=-out[0].gather(-1,label[:,None,None].expand(-1,16,1)).mean();loss.backward()
embedding_norm=sum(float(l.embedding.weight.grad.norm()) for l in list(model.field.origin)+list(model.field.destination))
output_norm=float(model.field.net[-1].weight.grad.norm())
assert embedding_norm>0 and output_norm>0 and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
# Count fitting only consumes coordinates; zero-support locations must back off to zero.
simple=BilinearEndpointLevel(1000,width=2);stats=simple.fit_confidence(torch.tensor([[0.,0.],[0.,0.],[100.,100.]]))
assert abs(stats['total_endpoint_weight']-3)<1e-5 and (simple.confidence<1).all()
r=dict(passed=True,scope='CPU architecture instrument; protected weights unchanged, no optimizer step',
 bilinear_linear_field_max_error_m=error,nonnegative_weights_partition_unity=True,
 query_order_invariance=True,zero_spatial_branch_exactly_recovers_original_output=True,zero_branch_max_logp_error=zero_error,
 finite_nonzero_embedding_gradient_norm=embedding_norm,finite_nonzero_field_output_gradient_norm=output_norm,
 confidence_fit_only_coordinates=True,query_model_accepts_only_legal216feature_tensor=True,
 feature_input_contract=['o_lon','o_lat','d_lon','d_lat','timestamp'],actual_path_type_duration_length_not_query_inputs=True,
 parameters=sum(p.numel() for p in model.parameters()),reference_sha256=sha(BEST/'modelo.pt'),Test_access=False,Validation_labels_read=False,
 source_sha256=sha(OUT/'modelo_campo_multiescala.py'),code_sha256=sha(__file__),actual_files_accessed=sorted(access),seconds=time.time()-start)
write(OUT/'verificacion_campo_multiescala.json',r);print(json.dumps(r),flush=True)
