"""Certify full saved point predictions and raw-five-field inference of a learned median."""
from base_replanteo import *
import torch,shutil,argparse
from inferencia import feature_input
from inferencia_mediana_directa import load_frozen_direct,predict_points
p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
access=deny_test();start=time.time();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction(.4)
model,ck,r,run,library,helper=load_frozen_direct(args.run);assert r['learned_update_selected']
artifact=run/'certificacion';artifact.mkdir(exist_ok=False)
for name in ['certificar_mediana_directa.py','inferencia_mediana_directa.py']:shutil.copy2(OUT/name,artifact/name)
d,V,names=feature_input('validation',read(BEST/'resultado.json'));assert names==ck['features']
cb=dict(np.load(OUT/'banco_proyeccion_incierta_s0/validation_banco.npz'));hb=dict(np.load(OUT/'datos_prototipos_historicos_s0/validation_historicos.npz'))
z=dict(np.load(run/'validacion.npz'));previous=dict(np.load(OUT/r['config']['reference_id']/'validacion.npz'))
for key in ['row','timestamp']:
 assert np.array_equal(d[key],z[key]) and np.array_equal(d[key],cb[key]) and np.array_equal(d[key],hb[key]) and np.array_equal(d[key],previous[key])
truth=np.load(DATA/'validation_y.npy');assert len(truth)==127941 and np.array_equal(truth,z['true']) and np.array_equal(truth,previous['true'])
point,continuous=predict_points(model,V,cb['bank'],hb['history'])
assert np.array_equal(point,z['pred']);metrics=metric(truth,point);assert metrics==r['validation']
continuous_error=float(np.max(np.abs(continuous-z['continuous_pred'])));assert continuous_error<.001
ii=np.sort(np.random.default_rng(20261122).choice(len(V),65,replace=False));keys=['o_lon','o_lat','d_lon','d_lat','timestamp']
items=[{k:int(d[k][i]) if k=='timestamp' else float(d[k][i]) for k in keys} for i in ii]
X,RC,RH,rnames=helper.raw_inputs(items,ck['base_config'],library)
assert rnames==names and np.max(np.abs(X-V[ii]))<1e-5 and np.array_equal(RC,cb['bank'][ii]) and np.array_equal(RH,hb['history'][ii])
raw,raw_continuous=predict_points(model,X,RC,RH);assert np.array_equal(raw,point[ii]);assert np.max(np.abs(raw_continuous-continuous[ii]))<.001
RX,RR,RHrev,_=helper.raw_inputs(items[::-1],ck['base_config'],library)
reverse,reverse_continuous=predict_points(model,RX,RR,RHrev)
SX,SC,SH,_=helper.raw_inputs([items[0]],ck['base_config'],library)
single,single_continuous=predict_points(model,SX,SC,SH)
assert np.array_equal(raw,reverse[::-1]) and np.array_equal(raw[:1],single)
assert np.max(np.abs(raw_continuous-reverse_continuous[::-1]))<.001 and np.max(np.abs(raw_continuous[:1]-single_continuous))<.001
np.savez_compressed(artifact/'predicciones_reproducidas_Validation.npz',row=d['row'],timestamp=d['timestamp'],pred=point,continuous_pred=continuous)
write(artifact/'consultas_raw_sin_etiquetas.json',items)
result=dict(passed=True,checkpoint_sha256=r['model_sha256'],reference_sha256=r['config']['reference_sha256'],validation=metrics,
 full127941saved_point_predictions_exact=True,continuous_max_error_s=continuous_error,
 raw65_global_feature_error=float(np.max(np.abs(X-V[ii]))),raw65_public_history_features_exact=True,
 raw65_points_match_full_saved=True,reverse_singleton_points_exact=True,continuous_order_tolerance_s=.001,
 information=keys,uncertainty='Point readout only; no new interval calibration or probability claim.',
 paired_vs_reference=paired(truth,point,previous['pred'],d['timestamp']),
 sources={n:sha(artifact/n) for n in ['certificar_mediana_directa.py','inferencia_mediana_directa.py']},
 reproduced_predictions_sha256=sha(artifact/'predicciones_reproducidas_Validation.npz'),
 Test_access=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(artifact/'resultado.json',result)
print(json.dumps(dict(passed=True,validation=metrics,raw65_reverse_singleton_exact=True,full_predictions_exact=True,seconds=result['seconds'])),flush=True)
