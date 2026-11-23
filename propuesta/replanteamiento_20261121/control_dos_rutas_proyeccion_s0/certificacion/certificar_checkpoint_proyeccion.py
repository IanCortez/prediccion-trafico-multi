"""Full Validation reproduction and raw-query independence of a frozen candidate."""
from base_replanteo import *
import torch,argparse,shutil
from inferencia import feature_input
from inferencia_proyeccion import load_frozen,raw_inputs,iter_outputs,predict_quantiles
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
access=deny_test();start=time.time();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction(.4)
model,ck,r,run=load_frozen(a.run);artifact=run/'certificacion';artifact.mkdir(exist_ok=False)
for name in ['certificar_checkpoint_proyeccion.py','inferencia_proyeccion.py']:shutil.copy2(OUT/name,artifact/name)
ref=dict(np.load(BEST/'validacion.npz'));z=dict(np.load(run/'validacion.npz'))
d,V,names=feature_input('validation',read(BEST/'resultado.json'));assert names==ck['features']
bank=dict(np.load(OUT/'banco_proyeccion_incierta_s0/validation_banco.npz'));C=bank['bank']
for key in ['row','timestamp']:assert np.array_equal(bank[key],d[key]) and np.array_equal(z[key],d[key]) and np.array_equal(z[key],ref[key])
truth=np.load(DATA/'validation_y.npy');assert np.array_equal(truth,z['true']) and np.array_equal(truth,ref['true'])
quantiles=[];gates=[];pit=[];nll_sum=0.;rps_sum=0.;rng=np.random.default_rng(20261122)
for first,out in iter_outputs(model,V,C):
 probability=out[0].exp().mean(1);n=len(probability);t=torch.as_tensor(truth[first:first+n],device=probability.device);label=((t-60)/15).long()
 mass=probability.gather(-1,label[:,None])[:,0];cdf=probability.cumsum(-1)
 nll_sum+=float(-mass.clamp_min(1e-12).double().log().sum())
 desired=(model.duration.duration_grid[None]>=t[:,None]).float();rps_sum+=float((cdf-desired).double().square().mean(-1).sum())
 lower=cdf.gather(-1,label[:,None])[:,0]-mass;u=torch.as_tensor(rng.uniform(size=n),device=probability.device)
 pit.append((lower+u*mass).cpu().numpy())
 quantiles.append(model.duration.quantiles(out[0]).cpu().numpy());gates.append(out[1][...,8:].exp().sum(-1).mean(1).cpu().numpy())
q=np.concatenate(quantiles);gate=np.concatenate(gates);pit=np.concatenate(pit)
assert np.array_equal(q[:,1],z['pred']);m=metric(truth,q[:,1])
assert m==r['validation']
ii=np.sort(np.random.default_rng(20261122).choice(len(V),65,replace=False));keys=['o_lon','o_lat','d_lon','d_lat','timestamp']
items=[{k:int(d[k][i]) if k=='timestamp' else float(d[k][i]) for k in keys} for i in ii]
X,RC,rnames=raw_inputs(items,ck['parent_config']);assert rnames==names and np.max(np.abs(X-V[ii]))<1e-5 and np.array_equal(RC,C[ii])
rq=predict_quantiles(model,X,RC);assert np.array_equal(rq[:,1],q[ii,1])
rx,rc,_=raw_inputs(items[::-1],ck['parent_config']);reverse=predict_quantiles(model,rx,rc)
sx,sc,_=raw_inputs([items[0]],ck['parent_config']);single=predict_quantiles(model,sx,sc)
assert np.array_equal(rq,reverse[::-1]) and np.array_equal(rq[:1],single)
np.savez_compressed(artifact/'intervalos_y_puerta_Validation.npz',row=d['row'],timestamp=d['timestamp'],quantiles=q,candidate_gate=gate,pit=pit.astype(np.float32))
write(artifact/'consultas_raw_sin_etiquetas.json',items)
out=dict(passed=True,checkpoint_sha256=r['model_sha256'],frozen_epoch=r['best_epoch'],
 full127941point_predictions_exact=True,validation=m,
 raw65_original216features_max_error=float(np.max(np.abs(X-V[ii]))),raw65_candidate_features_exact=True,
 raw65_medians_match_full_saved_predictions=True,reverse_and_singleton_all_quantiles_exact=True,
 query_fields=keys,uncertainty=dict(ensemble_NLL=nll_sum/len(truth),normalized_RPS=rps_sum/len(truth),
 coverage80=float(np.mean((truth>=q[:,0])&(truth<=q[:,2]))),mean_width80_s=float(np.mean(q[:,2]-q[:,0])),
 mean_candidate_gate=float(gate.mean()),candidate_gate_quantiles=np.quantile(gate,[.1,.5,.9]),
 randomized_PIT_decile_counts=np.histogram(pit,bins=np.linspace(0,1,11))[0],
 interpretation='Descriptive calibration on reusedValidation; no confirmation or nominal coverage guarantee.'),
 paired_vs_original=paired(truth,q[:,1],ref['pred'],d['timestamp']),
 reproduction_sources={name:sha(artifact/name) for name in ['certificar_checkpoint_proyeccion.py','inferencia_proyeccion.py']},
 artifact_sha256=sha(artifact/'intervalos_y_puerta_Validation.npz'),Test_access=False,
 actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(artifact/'resultado.json',out)
print(json.dumps(dict(passed=True,run=a.run,validation=m,full_prediction_reproduction_exact=True,raw65_reverse_singletons_exact=True,seconds=out['seconds'])),flush=True)
