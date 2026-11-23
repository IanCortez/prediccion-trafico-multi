"""Semantic proof against originalteacher and independent partialinterval reconstruction, CPUonly."""
from base_replanteo import *
import torch
from tokens_subtrayectos import SubtripTokenBank,query_parts_sub
from tokens_subtrayectos_arco import SubtripArcLengthBank
access=deny_test();start=time.time();torch.set_num_threads(2)
old=SubtripTokenBank('cpu');arc=SubtripArcLengthBank(shared=old);d=dict(np.load(DATA/'train_x.npz'));part=query_parts_sub(d)
z=dict(np.load(OUT/'auditoria_semantica_geometria_Train512.npz'));ii=z['train_index'];parents=np.full((len(ii),64),-1,np.int32);parents[:,0]=ii
first=np.zeros_like(parents,np.uint8);last=first.copy();last[:,0]=old.n.numpy()[ii]-1
q=[torch.as_tensor(part[key][ii]) for key in ['xy_m','hour','weekend']]
a=arc.features(*q,parents,first,last).float().numpy()[:,0];b=old.features(*q,parents,first,last).float().numpy()[:,0]
error=np.linalg.norm((a[:,:14]-z['original_teacher']).reshape(-1,7,2),axis=2)*1000
assert error.max()<5,error.max()
assert np.array_equal(a[:,16:],b[:,16:]),'Onlygeometry/path/circuity maychange'
path_relative=(a[:,14]*1000-z['original_full_length_m'])/z['original_full_length_m'];assert np.max(np.abs(path_relative))<.001
# Original semantic contract must ignore repeatedstationaryGPSsamples.
def toy(poly):
 obj=SubtripArcLengthBank.__new__(SubtripArcLengthBank);obj.__dict__.update(old.__dict__)
 obj.xy=torch.as_tensor(poly,dtype=torch.float32);obj.offset=torch.tensor([0,len(poly)])
 obj.n=torch.tensor([len(poly)]);obj.ts=torch.tensor([1390000000]);dist=np.r_[0,np.cumsum(np.linalg.norm(np.diff(poly,axis=0),axis=1))]
 obj.prefix_m=torch.as_tensor(dist,dtype=torch.float32)
 p=np.full((1,64),-1,np.int32);p[0,0]=0;f=np.zeros_like(p,np.uint8);j=f.copy();j[0,0]=len(poly)-1
 raw=obj.features(torch.tensor([[0.,0.,1000.,0.]]),torch.tensor([9.]),torch.tensor([False]),p,f,j)
 return raw[0,0].float().numpy()
poly=np.array([[0,0],[250,250],[500,500],[750,250],[1000,0]],float)
ta=toy(poly);tb=toy(np.repeat(poly,[5,1,1,1,4],axis=0))
assert np.array_equal(ta[:16],tb[:16]) and ta[16]!=tb[16]
# Actual crossfit sources with partial starts/ends, independent64-bit NumPy arclength geometry.
indices=dict(np.load(OUT/'indices_subtrayectos_s0/train_indices.npz'));eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index']
take=np.sort(np.random.default_rng(721).choice(eligible,256,replace=False))
p=indices['parent'][take];f=indices['first'][take];j=indices['last'][take]
qs=[torch.as_tensor(part[key][take]) for key in ['xy_m','hour','weekend']]
actual=arc.features(*qs,p,f,j).float().numpy();old_raw=old.features(*qs,p,f,j).float().numpy()
assert np.array_equal(actual[:,:,16:],old_raw[:,:,16:])
reverse=arc.features(*[v.flip(0) for v in qs],p[::-1].copy(),f[::-1].copy(),j[::-1].copy()).float().numpy()
single=arc.features(*[v[:1] for v in qs],p[:1],f[:1],j[:1]).float().numpy()
assert np.array_equal(actual,reverse[::-1]) and np.array_equal(actual[:1],single)
xy=old.xy.numpy();offset=old.offset.numpy();expected=np.zeros(actual.shape[:2]+(16,),float);u=np.arange(1,8)/8
valid=p>=0
for n in range(len(take)):
 qxy=part['xy_m'][take[n]].astype(float);qd=qxy[2:]-qxy[:2];qn=np.linalg.norm(qd)
 for k,parent in enumerate(p[n]):
  if parent<0:continue
  points=xy[offset[parent]+int(f[n,k]):offset[parent]+int(j[n,k])+1].astype(float)
  sd=points[-1]-points[0];sn=np.linalg.norm(sd);scale=np.clip(qn/sn,.5,2)
  co=np.dot(qd,sd)/(qn*sn);si=(qd[1]*sd[0]-qd[0]*sd[1])/(qn*sn)
  cumulative=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))];total=cumulative[-1]
  good=np.r_[True,np.diff(cumulative)>1e-7];middle=np.column_stack([np.interp(u*total,cumulative[good],points[good,c]) for c in range(2)])
  residual=middle-points[0]-u[:,None]*sd;rot=np.array([[co,-si],[si,co]])
  shape=(residual@rot.T*scale/1000).reshape(-1);path=total*scale/1000
  expected[n,k]=np.r_[shape,path,np.log(max(.01,path/(qn/1000)))]
scaled=np.abs(actual[:,:,:16].astype(float)-expected)/(1+np.abs(expected))
assert scaled.max()<.003,scaled.max()
result=dict(passed=True,wholeTrain_parent_teacher_n=512,
 scope='Source=query onlyfor originalTrain geometryidentityproof; actualmodelindices256queries areweekfoldcrossfit. No modeltraining or forecastmetric.',
 arclength_teacher_error_m_quantiles=np.quantile(error,[0,.25,.5,.75,.9,1]),max_relative_path_error=float(np.abs(path_relative).max()),
 partial_crossfit_Train_queries=256,partial_source_intervals=int(valid.sum()),independent_partial_max_scaled_error=float(scaled.max()),
 all_columns16to33_exactly_unchanged=True,stationary_sample_insertion_geometry_invariant=True,
 stationary_source_time_feature_correctly_changes=True,actual_source_reverse_singleton_exact=True,
 query_inputs_unchanged=True,source_sha256=sha(__file__),token_sha256=sha(OUT/'tokens_subtrayectos_arco.py'),
 prefix_sha256=read(OUT/'distancia_arco_puntos_Train_s0/resultado.json')['outputs_sha256'],
 no_Validation_or_Test=True,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(OUT/'verificacion_tokens_arco_Train.json',result);print(json.dumps(read(OUT/'verificacion_tokens_arco_Train.json')),flush=True)
