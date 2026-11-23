"""Repeat the current protected raw64attention prediction proof without replacing certification."""
from base_replanteo import *
import torch,argparse
from inferencia import feature_input
from inferencia_atencion_historia import load_frozen_attention,predict_quantiles,raw_inputs
ap=argparse.ArgumentParser();ap.add_argument('--run',default='atencion_historia_individual_s0');a=ap.parse_args()
access=deny_test();start=time.time();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction(.5)
model,ck,r,run,proto,library,helper,token=load_frozen_attention(a.run)
cert=read(run/'certificacion/resultado.json');assert cert['passed'] and cert['checkpoint_sha256']==r['model_sha256']
expected=dict(np.load(run/'certificacion/cuantiles_y_masas_Validation.npz'))
d,V,names=feature_input('validation',read(BEST/'resultado.json'));assert names==ck['features']
C=np.load(OUT/'banco_proyeccion_incierta_s0/validation_banco.npz')['bank'];H=np.load(OUT/'datos_prototipos_historicos_s0/validation_historicos.npz')['history']
qz=dict(np.load(OUT/'datos_historia_individual_s0/validation_indices.npz'))
for key in ['row','timestamp']:assert np.array_equal(d[key],expected[key]) and np.array_equal(d[key],qz[key])
query={key:qz[key] for key in ['xy_km','hour','weekend','indices']}
quantiles=predict_quantiles(model,library,V,C,H,query);assert np.array_equal(quantiles,expected['quantiles'])
saved=dict(np.load(run/'validacion.npz'));assert np.array_equal(quantiles[:,1],saved['pred'])
metrics=metric(saved['true'],quantiles[:,1]);assert metrics==r['validation']
items=read(run/'certificacion/consultas_raw_sin_etiquetas.json');ii=np.sort(np.random.default_rng(20261122).choice(len(V),65,replace=False))
X,RC,RH,RQ,_=raw_inputs(items,ck['base_config'],proto,library,helper,token)
raw=predict_quantiles(model,library,X,RC,RH,RQ);assert np.array_equal(raw,quantiles[ii])
result=dict(passed=True,validation=metrics,all127941quantiles_exact=True,raw65all_quantiles_exact=True,
 branch_checkpoint_sha256=r['model_sha256'],parent_checkpoint_sha256=r['config']['parent_sha256'],
 source_sha256=sha(__file__),Test_access=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(OUT/'reproduccion_atencion_historia.json',result)
print(json.dumps(dict(passed=True,validation=metrics,full_and_raw_quantiles_exact=True,seconds=result['seconds'])),flush=True)
