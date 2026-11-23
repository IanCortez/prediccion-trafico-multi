"""Full-cohort and raw5-field reproduction of a frozen historical-mixture candidate."""
from base_replanteo import *
import torch,argparse,shutil
from inferencia import feature_input
from inferencia_historica_fresca import load_frozen_history,raw_inputs,iter_outputs,predict_quantiles
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
access=deny_test();start=time.time();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction(.4)
model,ck,r,run,library=load_frozen_history(a.run);artifact=run/'certificacion';artifact.mkdir(exist_ok=False)
for name in ['certificar_historica_fresca.py','inferencia_historica_fresca.py']:shutil.copy2(OUT/name,artifact/name)
d,V,names=feature_input('validation',read(BEST/'resultado.json'));assert names==ck['features']
cb=dict(np.load(OUT/'banco_proyeccion_incierta_s0/validation_banco.npz'));hb=dict(np.load(OUT/'datos_prototipos_historicos_s0/validation_historicos.npz'))
z=dict(np.load(run/'validacion.npz'));original=dict(np.load(BEST/'validacion.npz'))
for key in ['row','timestamp']:assert np.array_equal(d[key],cb[key]) and np.array_equal(d[key],hb[key]) and np.array_equal(d[key],z[key]) and np.array_equal(d[key],original[key])
truth=np.load(DATA/'validation_y.npy');assert len(truth)==127941 and np.array_equal(truth,z['true']) and np.array_equal(truth,original['true'])
C=cb['bank'];H=hb['history'];qparts=[];massparts=[];pitparts=[];nll=0.;rps=0.;rng=np.random.default_rng(20261122)
for first,out in iter_outputs(model,V,C,H):
 prob=out[0].exp().mean(1);n=len(prob);t=torch.as_tensor(truth[first:first+n],device=prob.device);label=((t-60)/15).long()
 mass=prob.gather(-1,label[:,None])[:,0];cdf=prob.cumsum(-1)
 nll+=float(-mass.clamp_min(1e-12).double().log().sum())
 desired=(model.duration.duration_grid[None]>=t[:,None]).float();rps+=float((cdf-desired).double().square().mean(-1).sum())
 u=torch.as_tensor(rng.uniform(size=n),device=prob.device);pitparts.append((cdf.gather(-1,label[:,None])[:,0]-mass+u*mass).cpu().numpy())
 qparts.append(model.duration.quantiles(out[0]).cpu().numpy())
 w=out[1].exp().mean(1);massparts.append(torch.stack([w[:,:8].sum(-1),w[:,8:16].sum(-1),w[:,16:].sum(-1)],-1).cpu().numpy())
q=np.concatenate(qparts);groups=np.concatenate(massparts);pit=np.concatenate(pitparts)
assert np.array_equal(q[:,1],z['pred']);metrics=metric(truth,q[:,1]);assert metrics==r['validation']
ii=np.sort(np.random.default_rng(20261122).choice(len(V),65,replace=False));keys=['o_lon','o_lat','d_lon','d_lat','timestamp']
items=[{k:int(d[k][i]) if k=='timestamp' else float(d[k][i]) for k in keys} for i in ii]
X,RC,RH,rnames=raw_inputs(items,ck['base_config'],library);assert rnames==names
assert np.max(np.abs(X-V[ii]))<1e-5 and np.array_equal(RC,C[ii]) and np.array_equal(RH,H[ii])
rq=predict_quantiles(model,X,RC,RH);assert np.array_equal(rq[:,1],q[ii,1])
RX,RR,RHrev,_=raw_inputs(items[::-1],ck['base_config'],library);reverse=predict_quantiles(model,RX,RR,RHrev)
SX,SC,SH,_=raw_inputs([items[0]],ck['base_config'],library);single=predict_quantiles(model,SX,SC,SH)
assert np.array_equal(rq,reverse[::-1]) and np.array_equal(rq[:1],single)
np.savez_compressed(artifact/'intervalos_y_masas_Validation.npz',row=d['row'],timestamp=d['timestamp'],quantiles=q,mode_mass_general_public_historical=groups,pit=pit.astype(np.float32))
write(artifact/'consultas_raw_sin_etiquetas.json',items)
result=dict(passed=True,checkpoint_sha256=r['model_sha256'],frozen_epoch=r['best_epoch'],validation=metrics,
 full127941saved_point_predictions_exact=True,raw65_global_feature_error=float(np.max(np.abs(X-V[ii]))),
 raw65_public_and_history_features_exact=True,raw65_medians_match_full_saved=True,reverse_and_singleton_all_quantiles_exact=True,
 information=keys,uncertainty=dict(ensemble_NLL=nll/len(truth),normalized_RPS=rps/len(truth),
 coverage80=float(np.mean((truth>=q[:,0])&(truth<=q[:,2]))),mean_width80_s=float(np.mean(q[:,2]-q[:,0])),
 mean_mode_mass_general_public_historical=groups.mean(0),randomized_PIT_decile_counts=np.histogram(pit,bins=np.linspace(0,1,11))[0],
 interpretation='Descriptive reusedValidation calibration; not confirmation or a coverage guarantee.'),
 paired_vs_original=paired(truth,q[:,1],original['pred'],d['timestamp']),
 artifacts_sha256=sha(artifact/'intervalos_y_masas_Validation.npz'),
 sources={s:sha(artifact/s) for s in ['certificar_historica_fresca.py','inferencia_historica_fresca.py']},
 Test_access=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(artifact/'resultado.json',result)
print(json.dumps(dict(passed=True,run=a.run,validation=metrics,full_prediction_reproduction_exact=True,raw65_reverse_singletons_exact=True,seconds=result['seconds'])),flush=True)
