"""Independentcompleteheld replay; originalGPSpointbank is forbidden in this process."""
from base_replanteo import *
import torch,gc
from inferencia import feature_input
from modelo_puente_temporal_2d import TemporalBridge2D,canonical_predict
from datos_mezcla_individual_Train import build_public206
from puente_temporal_2d import score
start=time.time();access=deny_test()
forbidden=(OUT/'memoria_puntos_Train_s0/puntos_Train.npz').resolve()
attempts=[]
def no_original_points(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
  try:p=Path(os.fsdecode(args[0])).resolve()
  except (OSError,ValueError):return
  if p==forbidden:
   attempts.append(str(p));raise AssertionError('Predictionreplay may not read privateoriginalpointbank')
sys.addaudithook(no_original_points)
run=OUT/'puente_temporal_2d_Train_s0';r=read(run/'resultado.json');assert r['complete']
assert sha(run/'predicciones_held_Train.npz')==r['predictions_sha256']
assert sha(run/'constructor.pt')==r['constructor_sha256'] and sha(run/'protocolo.json')==r['protocol_sha256']
for name,digest in r['sources'].items():assert sha(run/'fuentes'/name)==digest and sha(OUT/name)==digest
for name,digest in read(run/'fuentes_publicas.json').items():assert sha(FROZEN/name)==digest
torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(.5)
meta=torch.load(run/'constructor.pt',map_location='cpu',weights_only=False);nfit=meta['fit_n']
source=OUT/'indices_escala_Train_s0/metadatos_Train.npz';assert sha(source)==meta['fit_metadata_sha256']
with np.load(source) as z:
 index=z['train_index'][nfit:];rows=z['row'][nfit:];timestamps=z['timestamp'][nfit:]
saved=dict(np.load(run/'predicciones_held_Train.npz'))
assert np.array_equal(saved['row'],rows) and np.array_equal(saved['timestamp'],timestamps)
d,X,names=feature_input('train',read(BEST/'resultado.json'))
keep=np.array([not name.startswith('mm_') for name in names]);names=[n for n,k in zip(names,keep) if k]
X=np.ascontiguousarray(X[index][:,keep]);assert names==meta['features'] and X.shape==(10000,206)
assert np.array_equal(d['row'][index],rows)
queries={key:d[key][index] for key in ['o_lon','o_lat','d_lon','d_lat','timestamp']}
probe=np.sort(np.random.default_rng(20261201).choice(10000,65,replace=False))
items=[{key:int(value[i]) if key=='timestamp' else float(value[i]) for key,value in queries.items()} for i in probe]
raw,names2=build_public206(items,read(BEST/'resultado.json')['config'])
assert names2==names and np.array_equal(raw,X[probe])
gx=torch.tensor(X,device='cuda');rx=torch.tensor(raw,device='cuda');checks={}
for item in r['checkpoints']:
 mode=item['arm'];file=Path(item['file']).resolve()
 assert file.is_relative_to(run.resolve()) and sha(file)==item['sha256']
 checkpoint=torch.load(file,map_location='cpu',weights_only=False)
 model=TemporalBridge2D(**checkpoint['constructor']).cuda();model.load_state_dict(checkpoint['state']);model.eval()
 lp,q=canonical_predict(model,gx);logp=lp.numpy();quant=q.numpy()
 arm=dict(np.load(run/mode/'predicciones_held_Train.npz'))
 assert sha(run/mode/'predicciones_held_Train.npz')==item['predictions_sha256']
 for key in ['row','timestamp','true']:assert np.array_equal(arm[key],saved[key])
 assert np.array_equal(quant,saved[mode+'_quantiles']) and np.array_equal(quant,arm['quantiles'])
 assert np.array_equal(logp.astype(np.float32),arm['log_prob'])
 assert np.array_equal(quant[:,1],arm['pred'])
 computed=metric(saved['true'],quant[:,1]);assert computed==r['metrics'][mode]
 target=torch.tensor((saved['true']/15).astype(np.int64));_,crps,nll=score(lp,target)
 for key,value in [('CRPS_s',float(crps.mean())),('bin_NLL',float(nll.mean()))]:
  assert abs(value-r['probability_scores'][mode][key])<1e-9
 rp,rq=canonical_predict(model,rx)
 reverse,reverseq=canonical_predict(model,rx[:17].flip(0));single,singleq=canonical_predict(model,rx[:1])
 assert torch.equal(rp,lp[probe]) and np.array_equal(rq.numpy(),quant[probe])
 assert torch.equal(rp[:17],reverse.flip(0)) and torch.equal(rp[:1],single)
 assert torch.equal(rq[:17],reverseq.flip(0)) and torch.equal(rq[:1],singleq)
 assert not attempts
 checks[mode]=dict(full10000quantiles_exact=True,full10000logprob_FP32_exact=True,raw65_probability_quantiles_exact=True,
  raw_reverse17_singleton_exact=True,computed_metrics=computed,original_pointbank_opened=False,model_sha256=item['sha256'])
 print(json.dumps(dict(stage='bridge_replay_arm_complete',arm=mode,metrics=computed,seconds=time.time()-start)),flush=True)
 del model,checkpoint,lp,q,rp,rq,reverse,single;gc.collect();torch.cuda.empty_cache()
proof=dict(passed=True,fit_n=nfit,held_n=10000,arms=checks,scope='Independentcompletepredictionreplay; not newprimaryevaluation.',
 raw_query=['o_lon','o_lat','d_lon','d_lat','timestamp'],raw65public206_exact=True,
 original_pointbank_open_attempts=attempts,no_true_queryN_or_trajectory_in_model=True,
 all_originalrows_labels_preserved=True,Validation=False,Test=False,seconds=time.time()-start,
 actual_data_files_accessed=sorted(access),source_sha256=sha(__file__))
write(run/'reproduccion_independiente.json',proof);print(json.dumps(proof),flush=True)
