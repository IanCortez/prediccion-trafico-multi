"""Reproduce full fixed-cohort points and raw-five-field queries for a frozen attention candidate."""
from base_replanteo import *
import torch,shutil,argparse
from inferencia import feature_input
from inferencia_atencion_historia import load_frozen_attention,raw_inputs,iter_attention,predict_quantiles
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
access=deny_test();start=time.time();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction(.5)
model,ck,r,run,proto,library,helper,token=load_frozen_attention(a.run)
artifact=run/'certificacion';artifact.mkdir(exist_ok=False)
for name in ['certificar_atencion_historia.py','inferencia_atencion_historia.py']:shutil.copy2(OUT/name,artifact/name)
d,V,names=feature_input('validation',read(BEST/'resultado.json'));assert names==ck['features']
cb=dict(np.load(OUT/'banco_proyeccion_incierta_s0/validation_banco.npz'));hb=dict(np.load(OUT/'datos_prototipos_historicos_s0/validation_historicos.npz'));qb=dict(np.load(OUT/'datos_historia_individual_s0/validation_indices.npz'))
z=dict(np.load(run/'validacion.npz'));previous=dict(np.load(OUT/r['config']['parent_id']/'validacion.npz'))
for key in ['row','timestamp']:
 assert np.array_equal(d[key],z[key]) and np.array_equal(d[key],cb[key]) and np.array_equal(d[key],hb[key]) and np.array_equal(d[key],qb[key]) and np.array_equal(d[key],previous[key])
truth=np.load(DATA/'validation_y.npy');assert len(truth)==127941 and np.array_equal(truth,z['true']) and np.array_equal(truth,previous['true'])
query={key:qb[key] for key in ['xy_km','hour','weekend','indices']};parts=[];masses=[];nll=0.;rps=0.
for first,n,out in iter_attention(model,library,V,cb['bank'],hb['history'],query):
 parts.append(model.duration.quantiles(out[0])[:n].cpu().numpy())
 prob=out[0][:n].exp().mean(1);target=torch.as_tensor(truth[first:first+n],device=prob.device);label=((target-60)/15).long()
 nll+=float(-prob.gather(-1,label[:,None])[:,0].clamp_min(1e-12).double().log().sum())
 desired=(model.duration.duration_grid[None]>=target[:,None]).float();rps+=float((prob.cumsum(-1)-desired).double().square().mean(-1).sum())
 w=out[1][:n].exp().mean(1);masses.append(torch.stack([w[:,:8].sum(-1),w[:,8:16].sum(-1),w[:,16:22].sum(-1),w[:,22:].sum(-1)],-1).cpu().numpy());del out
quantiles=np.concatenate(parts);mass=np.concatenate(masses);assert np.array_equal(quantiles[:,1],z['pred']);metrics=metric(truth,quantiles[:,1]);assert metrics==r['validation']
ii=np.sort(np.random.default_rng(20261122).choice(len(V),65,replace=False));keys=['o_lon','o_lat','d_lon','d_lat','timestamp']
items=[{k:int(d[k][i]) if k=='timestamp' else float(d[k][i]) for k in keys} for i in ii]
X,C,H,q,rnames=raw_inputs(items,ck['base_config'],proto,library,helper,token)
assert rnames==names and np.max(np.abs(X-V[ii]))<1e-5 and np.array_equal(C,cb['bank'][ii]) and np.array_equal(H,hb['history'][ii])
for key in query:assert np.array_equal(q[key],query[key][ii]),key
raw=predict_quantiles(model,library,X,C,H,q);assert np.array_equal(raw[:,1],quantiles[ii,1])
RX,RC,RH,RQ,_=raw_inputs(items[::-1],ck['base_config'],proto,library,helper,token)
reverse=predict_quantiles(model,library,RX,RC,RH,RQ)
SX,SC,SH,SQ,_=raw_inputs([items[0]],ck['base_config'],proto,library,helper,token)
single=predict_quantiles(model,library,SX,SC,SH,SQ)
assert np.array_equal(raw,reverse[::-1]) and np.array_equal(raw[:1],single)
np.savez_compressed(artifact/'cuantiles_y_masas_Validation.npz',row=d['row'],timestamp=d['timestamp'],quantiles=quantiles,mass_general_public_proto_individual=mass)
write(artifact/'consultas_raw_sin_etiquetas.json',items)
result=dict(passed=True,checkpoint_sha256=r['model_sha256'],parent_sha256=r['config']['parent_sha256'],validation=metrics,
 full127941saved_point_predictions_exact=True,raw65_cached_global_public_prototype_source_indices_queryparts_exact=True,
 raw65_points_match_full=True,reverse_singleton_all_quantiles_exact=True,
 uncertainty=dict(NLL=nll/len(truth),normalized_RPS=rps/len(truth),coverage80=float(np.mean((truth>=quantiles[:,0])&(truth<=quantiles[:,2]))),
 mean_width80_s=float(np.mean(quantiles[:,2]-quantiles[:,0])),mean_mass_general_public_proto_individual=mass.mean(0),
 interpretation='ReusedValidation descriptive calibration, not independent confirmation.'),
 paired_vs_parent=paired(truth,quantiles[:,1],previous['pred'],d['timestamp']),information=keys,
 sources={name:sha(artifact/name) for name in ['certificar_atencion_historia.py','inferencia_atencion_historia.py']},
 predictions_sha256=sha(artifact/'cuantiles_y_masas_Validation.npz'),Test_access=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(artifact/'resultado.json',result)
print(json.dumps(dict(passed=True,validation=metrics,full_point_reproduction_exact=True,raw65_reverse_singleton_all_quantiles_exact=True,seconds=result['seconds'])),flush=True)
