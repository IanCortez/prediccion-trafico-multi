"""Full saved-cohort and raw5 certification of a qualifying frozen streetvalue candidate."""
from base_replanteo import *
import torch,argparse,shutil
from inferencia import feature_input
from inferencia_valor_vial import load_frozen_street,raw_inputs,iter_street,predict_quantiles
ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args()
access=deny_test();start=time.time();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction(.5)
proof=read(OUT/'verificacion_proyecciones_valor_vial.json');assert proof['passed'] and proof['source_sha256']==sha(OUT/'proyecciones_valor_vial.py')
model,ck,r,run=load_frozen_street(a.run);artifact=run/'certificacion';assert not artifact.exists();artifact.mkdir()
for name in ['certificar_valor_vial.py','inferencia_valor_vial.py','proyecciones_valor_vial.py']:shutil.copy2(OUT/name,artifact/name)
d,V,names=feature_input('validation',read(BEST/'resultado.json'));assert names==ck['features']
z=dict(np.load(run/'validacion.npz'));truth=np.load(DATA/'validation_y.npy');bank=dict(np.load(OUT/'banco_valor_vial_s0/validation_proyecciones.npz'))
for key in ['row','timestamp']:assert np.array_equal(z[key],d[key]) and np.array_equal(bank[key],d[key])
assert len(truth)==127941 and np.array_equal(z['true'],truth)
bank={key:bank[key] for key in ['chain','fraction','error_m','remaining_km','reachable']}
parts=[];nll=0.;rps=0.
for first,n,lp in iter_street(model,V,bank):
 parts.append(model.quantiles(lp)[:n].cpu().numpy());prob=lp[:n].exp().mean(1);target=torch.as_tensor(truth[first:first+n],device=prob.device);label=((target-60)/15).long()
 nll+=float(-prob.gather(-1,label[:,None])[:,0].clamp_min(1e-12).double().log().sum())
 desired=(model.duration_grid[None]>=target[:,None]).float();rps+=float((prob.cumsum(-1)-desired).double().square().mean(-1).sum())
q=np.concatenate(parts);assert np.array_equal(q,z['quantiles']) and np.array_equal(q[:,1],z['pred']);assert metric(truth,q[:,1])==r['validation']
ii=np.sort(np.random.default_rng(20261122).choice(len(V),65,replace=False));fields=['o_lon','o_lat','d_lon','d_lat','timestamp']
items=[{key:int(d[key][i]) if key=='timestamp' else float(d[key][i]) for key in fields} for i in ii]
X,b,rnames=raw_inputs(items);assert rnames==names and np.max(np.abs(X-V[ii]))<1e-5
for key in bank:assert np.array_equal(b[key],bank[key][ii]),key
raw=predict_quantiles(model,X,b);assert np.array_equal(raw,q[ii])
RX,RB,_=raw_inputs(items[::-1]);rev=predict_quantiles(model,RX,RB)
SX,SB,_=raw_inputs(items[:1]);single=predict_quantiles(model,SX,SB)
assert np.array_equal(raw,rev[::-1]) and np.array_equal(raw[:1],single)
# Extra true-route/length/time keys are rejected by raw API; they cannot silently influence prediction.
for forbidden in ['route','length','duration','N','driver','call']:
 invalid=[dict(items[0],**{forbidden:777})]
 try:raw_inputs(invalid)
 except AssertionError:pass
 else:raise AssertionError('Forbidden query input accepted:'+forbidden)
np.savez_compressed(artifact/'cuantiles_Validation.npz',row=d['row'],timestamp=d['timestamp'],quantiles=q)
write(artifact/'consultas_raw_sin_etiquetas.json',items)
result=dict(passed=True,validation=r['validation'],checkpoint_sha256=r['model_sha256'],full127941_allquantiles_and_points_exact=True,
 raw65_globalfeatures_publicprojections_exact=True,raw65_quantiles_match_full=True,reverse_singleton_allquantiles_exact=True,extra_target_driver_call_keys_rejected=True,
 uncertainty=dict(NLL=nll/len(truth),normalized_RPS=rps/len(truth),coverage80=float(np.mean((truth>=q[:,0])&(truth<=q[:,2]))),width80_s=float(np.mean(q[:,2]-q[:,0])),
  interpretation='Descriptive onreusedValidation, not independentconfirmation.'),
 sources={name:sha(artifact/name) for name in ['certificar_valor_vial.py','inferencia_valor_vial.py','proyecciones_valor_vial.py']},
 predictions_sha256=sha(artifact/'cuantiles_Validation.npz'),Test_access=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(artifact/'resultado.json',result);print(json.dumps(dict(passed=True,validation=r['validation'],seconds=result['seconds'])),flush=True)
