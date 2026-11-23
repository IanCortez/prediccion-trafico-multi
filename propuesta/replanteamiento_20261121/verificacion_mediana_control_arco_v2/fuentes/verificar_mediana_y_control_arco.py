"""Independent finite differences and stationary-insertion control before any training."""
from base_replanteo import *
import torch,shutil,argparse,math
import torch.nn.functional as F
from cuantiles_mezcla_implicita import *
from tokens_transferencia_arco import ArcTransferBank,add_public_compatibility,compatibility_indices
from tokens_subtrayectos_arco import SubtripArcLengthBank
from modelo_mezcla_individual_mediana import IndividualSourceMixture,canonical_quantiles
from ruteo_proyectado import path_shape
ap=argparse.ArgumentParser();ap.add_argument('--version',default='v1');args=ap.parse_args()
access=deny_test();start=time.time();torch.set_num_threads(2)
run=OUT/('verificacion_mediana_control_arco_'+args.version);assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
sources=['verificar_mediana_y_control_arco.py','cuantiles_mezcla_implicita.py','tokens_transferencia_arco.py','modelo_mezcla_individual_mediana.py']
for name in sources:shutil.copy2(OUT/name,run/'fuentes'/name)
dtype=torch.float64
theta=torch.tensor([math.log(240),math.log(600),math.log(1050),-1.7,-1.2,-1.5,.2,-.3,.4],dtype=dtype,requires_grad=True)
def output(value):
 loc=value[:3][None,None];sigma=(.08+F.softplus(value[3:6]))[None,None];w=value[6:].log_softmax(0)[None,None]
 return loc,sigma,w
def scalar(value,kind):
 loc,sigma,w=output(value)
 if kind=='median':return mixture_quantile(loc,sigma,w,implicit=True,iterations=48).sum()
 if kind=='nll':return -bin_logprob(torch.tensor([[600.]],dtype=dtype),loc,sigma,w).sum()
 return mixture_loss((loc,sigma,w),torch.tensor([700.],dtype=dtype))[0]
gradients={}
for kind in ['median','nll','loss']:
 value=scalar(theta,kind);analytic=torch.autograd.grad(value,theta)[0].detach().numpy()
 fd=[]
 for i in range(len(theta)):
  plus=theta.detach().clone();minus=theta.detach().clone();plus[i]+=1e-5;minus[i]-=1e-5
  fd.append(float((scalar(plus,kind)-scalar(minus,kind))/2e-5))
 fd=np.asarray(fd);assert np.allclose(analytic,fd,rtol=8e-6,atol=3e-4),(kind,analytic,fd)
 gradients[kind]=dict(max_abs_error=float(np.max(np.abs(analytic-fd))),analytic=analytic,finite_difference=fd)
loc=torch.tensor([[math.log(600)]],dtype=dtype);sigma=torch.tensor([[.3]],dtype=dtype);w=torch.zeros_like(loc)
assert abs(float(mixture_quantile(loc,sigma,w,iterations=48))-600)<1e-8
grid=torch.arange(60,1800,15,dtype=dtype)
loc,sigma,w=output(theta.detach());lp=bin_logprob(grid[:,None],loc.expand(116,-1,-1),sigma.expand(116,-1,-1),w.expand(116,-1,-1))
assert abs(float(lp.exp().sum())-1)<3e-7
for where,bound in [(math.log(1),60),(math.log(10000),1785)]:
 assert float(mixture_quantile(torch.tensor([[where]],dtype=dtype),torch.tensor([[.01]],dtype=dtype),torch.zeros((1,1),dtype=dtype)))==bound
# Mixture median differs from the arithmetic average of membermedians.
mu=torch.tensor([[[math.log(100)],[math.log(900)]]],dtype=dtype);sig=torch.full_like(mu,.5);ww=torch.zeros_like(mu)
q=ensemble_quantiles((mu,sig,ww));assert abs(float(q[0,1])-300)<1e-8
assert abs(float(q[0,1])-500)>199
write(run/'gradientes_mediana.json',dict(passed=True,gradients=gradients,bin_mass_sum=float(lp.exp().sum()),mixture_median_not_average=True,
 reference='https://proceedings.neurips.cc/paper_files/paper/2018/file/92c8c96e4c37100777c7190b76d28233-Paper.pdf',
 interpretation='Own deterministic median adaptation of implicitCDF differentiation; no claim of methodological novelty for the derivative.'))
# Synthetic curvedsource, partialinterval, insert stationarysamples strictlyinside it.
n=40;t=np.linspace(0,1,n);points=np.column_stack([1500*t,180*np.sin(2*np.pi*t)]).astype(np.float32)
i=2;j=36;insert=20;extra=5
points2=np.concatenate([points[:insert],np.repeat(points[insert:insert+1],extra,axis=0),points[insert:]])
one=ArcTransferBank.from_arrays(points,[n],[1373000000]);two=ArcTransferBank.from_arrays(points2,[n+extra],[1373000000])
parent=np.full((1,64),-1,np.int32);parent[0,0]=0
first=np.zeros((1,64),np.uint8);last=np.zeros_like(first);first[0,0]=i;last[0,0]=j
last2=last.copy();last2[0,0]+=extra
xy=torch.tensor(np.r_[points[i],points[j]][None],dtype=torch.float32);hour=torch.tensor([12.]);weekend=torch.tensor([False])
actual=one.features(xy,hour,weekend,parent,first,last);actual2=two.features(xy,hour,weekend,parent,first,last2)
control=one.features(xy,hour,weekend,parent,first,last,geometry_proxy=True)
control2=two.features(xy,hour,weekend,parent,first,last2,geometry_proxy=True)
assert torch.equal(control,control2),float((control-control2).abs().max())
changed=(actual!=actual2).any((0,1)).nonzero().flatten().tolist();assert changed==[16,32],changed
old=SubtripArcLengthBank.features(one,xy,hour,weekend,parent,first,last)
old2=SubtripArcLengthBank.features(two,xy,hour,weekend,parent,first,last2)
assert float((old[...,28:30]-old2[...,28:30]).abs().max())>.001
# Compare native publicroad path_shape on the same physicalpolyline, including a rotated/scaledquery.
names=[v for v in read(BEST/'resultado.json')['features'] if not v.startswith('mm_')];assert len(names)==206
shape_idx,available_idx=compatibility_indices(names)
shape_errors=[]
for angle,scale in [(0.,1.),(.6,1.3)]:
 rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
 path=points[i:j+1].astype(float)@rotation.T*scale+np.array([210.,-80.])
 path=path[np.r_[True,np.linalg.norm(np.diff(path,axis=0),axis=1)>1e-7]]
 u=np.arange(len(path)-1,dtype=np.int32);v=u+1;length=np.linalg.norm(np.diff(path,axis=0),axis=1)
 result=np.zeros(26,np.float32)
 path_shape(path[0],path[-1],path[0],np.arange(len(u),dtype=np.int32),length,len(u),path,u,v,length,np.empty((len(path)+4,2)),np.empty(len(path)+4),result)
 qxy=torch.tensor(np.r_[path[0],path[-1]][None],dtype=torch.float32)
 raw=one.features(qxy,hour,weekend,parent,first,last)
 error=float(np.max(np.abs(raw[0,0,:14].numpy().astype(float)-result[:14])));shape_errors.append(error);assert error<.0003,error
 xx=torch.zeros((1,206))
 for indices in shape_idx:xx[:,indices]=torch.tensor(result[:14])[None]
 xx[:,available_idx]=1
 aug=add_public_compatibility(raw,xx,shape_idx,available_idx);assert aug.shape==(1,64,38)
 assert float(aug[0,0,34:36].abs().max())<.0003
write(run/'control_estacionario_y_geometria.json',dict(passed=True,insertion_points=extra,only_actual_time_columns_changed=changed,
 geometryproxy_all34columns_exact=True,old_index_fraction_changes_observed=True,publicpath_shape_maxerror_km=max(shape_errors),
 rotation_and_scale_physical_frame_verified=True,synthetic_only_no_queryGT=True))
# Actual architecture sanity and objective/inference agreement.
torch.manual_seed(20261125)
ctor=dict(xmean=np.zeros(206),xstd=np.ones(206),rmean=np.zeros(38),rstd=np.ones(38),anchor_indices=np.repeat([0,1,2,3],2),
 anchor_scale=np.ones(8),anchor_coef=np.column_stack([np.full(8,math.log(500)),np.full(8,.1)]),source_mode='observed')
model=IndividualSourceMixture(**ctor)
x=torch.randn(3,206);raw=torch.zeros((3,64,38));raw[...,27]=1;raw[...,16]=torch.linspace(200,1000,64).log();raw[...,26]=-torch.linspace(0,5,64);raw[...,32]=math.log(900)
y=torch.tensor([300.,600.,900.])
opt=torch.optim.AdamW(model.parameters(),lr=.001)
for step in range(2):
 model.train();opt.zero_grad(set_to_none=True);loss,mae,nll=mixture_loss(model(x,raw),y);assert torch.isfinite(loss);loss.backward()
 assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 opt.step()
assert any(p.grad is not None and float(p.grad.abs().sum())>0 for p in model.members[0].query.parameters())
model.eval();out=model(x,raw);loss,mae,nll=mixture_loss(out,y);pred=ensemble_quantiles(out)
assert torch.allclose(mae*600,(pred[:,1]-y).abs().mean(),atol=.0002,rtol=1e-6)
canonical=canonical_quantiles(model,x,raw);rev=canonical_quantiles(model,x.flip(0),raw.flip(0));single=canonical_quantiles(model,x[:1],raw[:1])
assert torch.equal(canonical,rev.flip(0)) and torch.equal(canonical[:1],single)
permutation=torch.randperm(64);permuted=canonical_quantiles(model,x,raw[:,permutation]);permutation_error=float((canonical-permuted).abs().max())
assert permutation_error<.002,permutation_error
empty=canonical_quantiles(model,x,torch.zeros_like(raw));assert torch.isfinite(empty).all() and torch.all(empty[:,0]<=empty[:,1]) and torch.all(empty[:,1]<=empty[:,2])
result=dict(passed=True,implicit_median_finite_differences=True,folded_bins_normalized=True,stationary_insertion_control_exact=True,
 source_time_only_explicit=True,publicroad_geometry_frame_proven=True,ensemble_training_median_matches_inference=True,
 query_and_source_gradients_finite=True,canonical256_query_reverse_singleton_exact=True,sourcepermutation_maxerror_s=permutation_error,
 empty_source_fallback_finite=True,source_features=38,members=4,components_per_member=72,
 sources={name:sha(run/'fuentes'/name) for name in sources},Validation=False,Test=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(run/'resultado.json',result);print(json.dumps(read(run/'resultado.json')),flush=True)
