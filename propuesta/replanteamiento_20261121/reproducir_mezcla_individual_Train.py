"""Independent originalprediction replay and signed-log-tail evaluation of identical weights."""
from base_replanteo import *
import torch,gc,shutil
from inferencia import feature_input
from modelo_mezcla_individual_mediana import IndividualSourceMixture,canonical_quantiles
from cuantiles_mezcla_balanceada import stable_canonical_quantiles
from tokens_transferencia_arco import ArcTransferBank
from datos_mezcla_individual_Train import tensor_tokens,build_public206
from recuperar_fuentes_ordenadas import OrderedSourceMemory
access=deny_test();start=time.time();run=OUT/'mezcla_individual_mediana_Train_s0';r=read(run/'resultado.json')
assert r['complete'] and sha(run/'predicciones_held_Train.npz')==r['predictions_sha256']
assert sha(run/'constructor.pt')==r['constructor_sha256']
for name,digest in r['sources'].items():assert sha(run/'fuentes'/name)==digest and sha(OUT/name)==digest
stableproof=read(OUT/'verificacion_cuantiles_balanceados_s0/resultado.json');assert stableproof['passed']
for name,digest in stableproof['sources'].items():assert sha(OUT/name)==digest
torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(.5)
saved=dict(np.load(run/'predicciones_held_Train.npz'))
with np.load(OUT/'oraculo_tiempo_calle_train_s0/auxiliares_Train.npz') as z:q={k:z[k] for k in ['row','train_index','timestamp']}
for key in ['row','timestamp']:assert np.array_equal(saved[key],q[key][40000:])
meta=torch.load(run/'constructor.pt',map_location='cpu',weights_only=False)
d,X,names=feature_input('train',read(BEST/'resultado.json'));keep=np.array([not name.startswith('mm_') for name in names]);names=[n for n,k in zip(names,keep) if k]
X=np.ascontiguousarray(X[q['train_index'][40000:]][:,keep]);assert names==meta['features']
dd={k:d[k][q['train_index'][40000:]] for k in ['o_lon','o_lat','d_lon','d_lat','timestamp']}
source=OUT/'piloto_cdf_ordenada_Train_s0';src=read(source/'resultado.json');assert sha(source/'indices_fuentes.npz')==src['sourceindices_sha256']
with np.load(source/'indices_fuentes.npz') as z:refs={k:z[k][40000:] for k in ['parent','first','last']}
bank=ArcTransferBank('cuda');tokens={key:torch.empty((10000,64,38),device='cuda',dtype=torch.float16) for key in ['observed','geometry']}
for a in range(0,10000,256):
 b=min(a+256,10000);queries={k:v[a:b] for k,v in dd.items()};rr={k:v[a:b] for k,v in refs.items()}
 for mode in ['observed','geometry']:
  tokens[mode][a:b]=tensor_tokens(bank,queries,X[a:b],**rr,shape_indices=meta['shape_indices'],available_indices=meta['available_indices'],geometry_proxy=mode=='geometry')
# Independent raw5 source retrieval and publicgeometry features for65heldqueries.
probe=np.sort(np.random.default_rng(20261126).choice(10000,65,replace=False))
items=[{k:int(v[i]) if k=='timestamp' else float(v[i]) for k,v in dd.items()} for i in probe]
rawX,rawnames=build_public206(items,read(BEST/'resultado.json')['config'])
assert rawnames==names and np.array_equal(rawX,X[probe]),float(np.max(np.abs(rawX-X[probe])))
memory=OrderedSourceMemory(source_arrival_before=meta['source_cutoff']);memory.build_tree(5)
rawdd={k:v[probe] for k,v in dd.items()};fresh=memory.query(rawdd)
for k,key in enumerate(['parent','first','last']):assert np.array_equal(fresh[k],refs[key][probe]),key
rawtokens={}
for mode in ['observed','geometry']:
 rawtokens[mode]=tensor_tokens(bank,rawdd,rawX,parent=fresh[0],first=fresh[1],last=fresh[2],shape_indices=meta['shape_indices'],available_indices=meta['available_indices'],geometry_proxy=mode=='geometry')
 assert torch.equal(rawtokens[mode],tokens[mode][probe])
del memory;gc.collect()
gx=torch.tensor(X,device='cuda');rx=torch.tensor(rawX,device='cuda')
results={};stable_predictions={};stable_metrics={}
for checkpoint in r['checkpoints']:
 mode=checkpoint['arm'];path=Path(checkpoint['file']).resolve();assert path.is_relative_to(run) and sha(path)==checkpoint['sha256']
 ck=torch.load(path,map_location='cpu',weights_only=False);model=IndividualSourceMixture(**ck['constructor']).cuda();model.load_state_dict(ck['state']);model.eval()
 raw=tokens['geometry' if mode=='geometry' else 'observed'];rt=rawtokens['geometry' if mode=='geometry' else 'observed']
 normal=canonical_quantiles(model,gx,raw).numpy();normal_error=float(np.max(np.abs(normal-saved[mode+'_quantiles'])))
 assert normal_error<=.001,(mode,normal_error)
 stable=stable_canonical_quantiles(model,gx,raw).numpy();stable_predictions[mode]=stable;stable_metrics[mode]=metric(saved['true'],stable[:,1])
 raw_normal=canonical_quantiles(model,rx,rt).numpy();raw_stable=stable_canonical_quantiles(model,rx,rt).numpy()
 assert np.max(np.abs(raw_normal-normal[probe]))<=.001
 assert np.max(np.abs(raw_stable-stable[probe]))<=.001
 reverse=stable_canonical_quantiles(model,rx[:17].flip(0),rt[:17].flip(0)).numpy()
 singleton=stable_canonical_quantiles(model,rx[:1],rt[:1]).numpy()
 assert np.array_equal(raw_stable[:17],reverse[::-1]) and np.array_equal(raw_stable[:1],singleton)
 difference=np.abs(stable[:,1]-normal[:,1])
 results[mode]=dict(original_full10000_quantiles_maxerror_s=normal_error,raw65_sourceindices_and_tokens_exact=True,
  stable_raw65_maxerror_s=float(np.max(np.abs(raw_stable-stable[probe]))),stable_reverse17_singleton_exact=True,
  stable_vs_naive_median_difference_s=dict(mean=float(difference.mean()),max=float(difference.max()),p99=float(np.quantile(difference,.99)),n_above1ms=int((difference>.001).sum())),
  stable_metrics=stable_metrics[mode],checkpoint_sha256=checkpoint['sha256'])
 print(json.dumps(dict(stage='independent_replay_complete',arm=mode,result=results[mode],seconds=time.time()-start)),flush=True)
 del model,ck;gc.collect();torch.cuda.empty_cache()
comparisons={key:paired(saved['true'],stable_predictions['observed'][:,1],other,saved['timestamp']) for key,other in
 [('vs_neuralbaseline',stable_predictions['none'][:,1]),('vs_geometryproxy',stable_predictions['geometry'][:,1]),('vs_CatBoostreference',saved['reference'])]}
for value in comparisons.values():value['interpretation']='Exploratory reusedTrain, stable mathematically equivalent quantile evaluation of identical frozenparameters; no confirmation.'
passed=all(comparisons[key]['delta_mae_s']<=-amount and comparisons[key]['ci95_s'][1]<0 for key,amount in
 [('vs_neuralbaseline',5),('vs_geometryproxy',2),('vs_CatBoostreference',5)])
np.savez_compressed(run/'predicciones_balanceadas_held_Train.npz',row=saved['row'],timestamp=saved['timestamp'],true=saved['true'],reference=saved['reference'],
 **{key:value[:,1] for key,value in stable_predictions.items()},**{key+'_quantiles':value for key,value in stable_predictions.items()})
result=dict(passed=True,scope='Independentfull10000originalquantile replay and raw65fivefield probes, all3fixed80epochcheckpoints.',
 arms=results,stable_metrics=stable_metrics,stable_comparisons=comparisons,stable_practical_gate_passed=bool(passed),
 stable_method='Signed logtail residual withFloat64, normalizedweights; adversarial symmetry and analyticfinite-difference checks passed before observedarmoutcome. No checkpoint retraining/selection or changedlabels.',
 stable_predictions_sha256=sha(run/'predicciones_balanceadas_held_Train.npz'),stable_code_sha256=sha(OUT/'cuantiles_mezcla_balanceada.py'),
 replay_source_sha256=sha(__file__),all_completed_original_metrics_and_predictions_retained=True,
 raw_API=['o_lon','o_lat','d_lon','d_lat','timestamp'],sourcearrival_and_weekcutoff_preserved=True,Validation_access=False,Test_access=False,
 actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(run/'reproduccion_independiente.json',result);print(json.dumps(read(run/'reproduccion_independiente.json')),flush=True)
