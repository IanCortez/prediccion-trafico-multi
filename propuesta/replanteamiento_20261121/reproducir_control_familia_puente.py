"""Independentall10000/raw65matchedcontrol replay; originalGPSpointbank is forbidden."""
from base_replanteo import *
import torch,gc
from inferencia import feature_input
from modelo_control_lognormal_puente import PublicLognormalControl,canonical_predict_control
from datos_mezcla_individual_Train import build_public206
from puente_temporal_2d import score
start=time.time();access=deny_test();attempts=[]
forbidden=(OUT/'memoria_puntos_Train_s0/puntos_Train.npz').resolve()
def no_points(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
  try:p=Path(os.fsdecode(args[0])).resolve()
  except (OSError,ValueError):return
  if p==forbidden:attempts.append(str(p));raise AssertionError('Controlinference cannot read originalpointbank')
sys.addaudithook(no_points)
run=OUT/'control_familia_puente_2d_Train_s0';r=read(run/'resultado.json');assert r['complete']
for name,digest in r['sources'].items():assert sha(run/'fuentes'/name)==digest and sha(OUT/name)==digest
for file,key in [('predicciones_held_Train.npz','predictions_sha256'),('constructor.pt','constructor_sha256'),('protocolo.json','protocol_sha256')]:assert sha(run/file)==r[key]
meta=torch.load(run/'constructor.pt',map_location='cpu',weights_only=False)
indexfile=OUT/'indices_escala_Train_s0/metadatos_Train.npz';assert sha(indexfile)==meta['shared_fit_metadata_sha256']
with np.load(indexfile) as z:index=z['train_index'][meta['fit_n']:]
saved=dict(np.load(run/'predicciones_held_Train.npz'))
d,X,names=feature_input('train',read(BEST/'resultado.json'))
keep=np.array([not name.startswith('mm_') for name in names]);names=[name for name,k in zip(names,keep) if k]
X=np.ascontiguousarray(X[index][:,keep]);assert names==meta['features'] and X.shape==(10000,206)
assert np.array_equal(d['row'][index],saved['row']) and np.array_equal(d['timestamp'][index],saved['timestamp'])
query={key:d[key][index] for key in ['o_lon','o_lat','d_lon','d_lat','timestamp']}
probe=np.sort(np.random.default_rng(20261202).choice(10000,65,replace=False))
items=[{key:int(value[i]) if key=='timestamp' else float(value[i]) for key,value in query.items()} for i in probe]
raw,names2=build_public206(items,read(BEST/'resultado.json')['config']);assert names2==names and np.array_equal(raw,X[probe])
file=Path(r['model_file']).resolve();assert file.is_relative_to(run.resolve()) and sha(file)==r['model_sha256']
torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(.5)
checkpoint=torch.load(file,map_location='cpu',weights_only=False)
model=PublicLognormalControl(**checkpoint['constructor']).cuda();model.load_state_dict(checkpoint['state']);model.eval()
lp,q=canonical_predict_control(model,torch.tensor(X,device='cuda'))
assert np.array_equal(lp.numpy().astype(np.float32),saved['log_prob']) and np.array_equal(q.numpy(),saved['quantiles'])
assert np.array_equal(q.numpy()[:,1],saved['pred']) and metric(saved['true'],saved['pred'])==r['metrics']
target=torch.tensor((saved['true']/15).astype(np.int64));_,crps,nll=score(lp,target)
assert abs(float(crps.mean())-r['probability_scores']['CRPS_s'])<1e-9
assert abs(float(nll.mean())-r['probability_scores']['bin_NLL'])<1e-9
gx=torch.tensor(raw,device='cuda');rawlp,rawq=canonical_predict_control(model,gx)
reverse,reverseq=canonical_predict_control(model,gx[:17].flip(0));single,singleq=canonical_predict_control(model,gx[:1])
assert torch.equal(rawlp,lp[probe]) and torch.equal(rawq,q[probe])
assert torch.equal(rawlp[:17],reverse.flip(0)) and torch.equal(rawlp[:1],single)
assert torch.equal(rawq[:17],reverseq.flip(0)) and torch.equal(rawq[:1],singleq)
history=read(run/'historia.json');assert len(history)==4 and all(row['train_n']==951718 and row['updates']==3718 for row in history)
for key,other in [('Student_vs_matched_lognormal',saved['pred']),('Student_vs_saved_publicNN',saved['saved_publicNN'])]:
 pair=paired(saved['true'],saved['Student_noaux'],other,saved['timestamp'])
 assert abs(pair['delta_mae_s']-r['comparisons'][key]['delta_mae_s'])<1e-12
assert not attempts
proof=dict(passed=True,scope='Independentcompleteheld10000probabilities/quantiles andraw65inference.',
 all10000probabilities_FP32_exact=True,all10000quantiles_exact=True,raw65probabilities_andquantiles_exact=True,
 raw_reverse17_singleton_exact=True,all_four_fulltrainingpasses_verified=True,pairedcontrasts_recomputed=True,
 original_pointbank_open_attempts=attempts,raw_query=['o_lon','o_lat','d_lon','d_lat','timestamp'],
 computed_metrics=r['metrics'],probability_scores=r['probability_scores'],Test=False,Validation=False,
 actual_data_files_accessed=sorted(access),seconds=time.time()-start,source_sha256=sha(__file__))
write(run/'reproduccion_independiente.json',proof);print(json.dumps(proof),flush=True)
