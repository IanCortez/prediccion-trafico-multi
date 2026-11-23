"""Compare source geometry against the ORIGINAL independently prepared Train arclength teacher."""
from base_replanteo import *
import torch
from tokens_subtrayectos import SubtripTokenBank,query_parts_sub
access=deny_test();start=time.time();torch.set_num_threads(2)
d=dict(np.load(DATA/'train_x.npz'));gt=dict(np.load(DATA/'geometria_train.npz'))
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index']
pos=np.searchsorted(gt['row'],d['row'][eligible]);safe=np.minimum(pos,len(gt['row'])-1)
available=(pos<len(gt['row']))&(gt['row'][safe]==d['row'][eligible]);pool=eligible[available]
ii=np.sort(np.random.default_rng(6881).choice(pool,512,replace=False));lookup=np.searchsorted(gt['row'],d['row'][ii])
bank=SubtripTokenBank('cpu');n=bank.n.numpy();offset=bank.offset.numpy();xy=bank.xy.numpy()
parents=np.full((len(ii),64),-1,np.int32);first=np.zeros_like(parents,np.uint8);last=first.copy()
parents[:,0]=ii;last[:,0]=n[ii]-1
parts=query_parts_sub(d);raw=bank.features(*[torch.as_tensor(parts[key][ii]) for key in ['xy_m','hour','weekend']],parents,first,last).float().numpy()[:,0]
u=np.arange(1,8)/8
def reference(p,arclength):
 p=np.asarray(p,float);segment=np.linalg.norm(np.diff(p,axis=0),axis=1)
 cumulative=np.r_[0,np.cumsum(segment)];total=cumulative[-1]
 if arclength:
  valid=np.r_[True,np.diff(cumulative)>1e-7];middle=np.column_stack([np.interp(u*total,cumulative[valid],p[valid,k]) for k in range(2)])
 else:
  middle=np.column_stack([np.interp(u*(len(p)-1),np.arange(len(p)),p[:,k]) for k in range(2)])
 straight=p[0]*(1-u[:,None])+p[-1]*u[:,None]
 return (middle-straight).reshape(-1)/1000,total
arc=[];length=[]
for parent in ii:
 shape,total=reference(xy[offset[parent]:offset[parent+1]],True);arc.append(shape);length.append(total)
arc=np.array(arc);length=np.array(length);expected=gt['X'][lookup,1:15].astype(float)
arc_error=np.linalg.norm((arc-expected).reshape(-1,7,2),axis=2)*1000
old_error=np.linalg.norm((raw[:,:14]-expected).reshape(-1,7,2),axis=2)*1000
assert arc_error.max()<.02,arc_error.max()
length_error=np.abs(length-gt['X'][lookup,0]);assert length_error.max()<.02,length_error.max()
toy=np.array([[0,0],[500,500],[1000,0]],float);repeated=np.repeat(toy,[7,1,4],axis=0)
a,_=reference(toy,True);b,_=reference(repeated,True);ta,_=reference(toy,False);tb,_=reference(repeated,False)
assert np.allclose(a,b,atol=1e-12) and not np.allclose(ta,tb)
np.savez_compressed(OUT/'auditoria_semantica_geometria_Train512.npz',row=d['row'][ii],train_index=ii,
 original_teacher=expected,source_time_sampled=raw[:,:14],source_arclength=arc,
 original_full_length_m=length,source_ninepoint_length_m=raw[:,14]*1000)
result=dict(complete=True,Train_source_parents=512,semantic_mismatch_verified=True,
 original_teacher='7normalizedarclengthdeviations fromODchord; fullpolyline length',
 tested_source='Current34Dsource uses7equalGPSindex/time points andninepointapproximatepathlength',
 old_point_error_m_quantiles=np.quantile(old_error,[0,.25,.5,.75,.9,1]),
 corrected_arclength_vs_original_teacher_max_error_m=float(arc_error.max()),
 full_polyline_length_vs_original_teacher_max_error_m=float(length_error.max()),
 old_path_length_relative_error_quantiles=np.quantile((raw[:,14]*1000-length)/length,[0,.25,.5,.75,.9,1]),
 stationary_insertion=dict(correct_arclength_invariant=True,current_time_sample_geometry_changes=True),
 interpretation='Semanticauxiliaryprior mismatch, not queryinformationleak or changedprimaryT. CurrentMAE129.990034 remains valid foritsfrozenmethod.',
 correction='Separatefuturetokens samplebymetricarclength andexactsourcepolyline length, retainALLtime/context fields andsameNNarchitecture; neverfeedcorrectedtokens intooldweights.',
 no_active_sources_or_weights_changed=True,no_Validation_data=True,Test_access=False,actual_data_files_accessed=sorted(access),
 source_sha256=sha(__file__),oldtoken_sha256=sha(OUT/'tokens_subtrayectos.py'),
 original_definition_sha256=sha(FROZEN/'preparar_geometria_aux.py'),seconds=time.time()-start)
write(OUT/'auditoria_semantica_geometria_subtrayectos.json',result)
print(json.dumps(read(OUT/'auditoria_semantica_geometria_subtrayectos.json')),flush=True)
