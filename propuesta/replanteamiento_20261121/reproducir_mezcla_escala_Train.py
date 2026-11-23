"""Independent fullheld/raw5 replay of the streamed all-data fixed4epoch experiment."""
from base_replanteo import *
import torch,gc
from inferencia import feature_input
from modelo_mezcla_individual_mediana import IndividualSourceMixture
from cuantiles_mezcla_balanceada import stable_canonical_quantiles
from tokens_transferencia_arco import ArcTransferBank
from datos_mezcla_individual_Train import tensor_tokens,build_public206
from recuperar_fuentes_ordenadas import OrderedSourceMemory
access=deny_test();start=time.time();run=OUT/'mezcla_individual_escala_Train_s0';result=read(run/'resultado.json');assert result['complete']
assert sha(run/'predicciones_held_Train.npz')==result['predictions_sha256'] and sha(run/'constructor.pt')==result['constructor_sha256']
for name,digest in result['sources'].items():assert sha(run/'fuentes'/name)==digest and sha(OUT/name)==digest
torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(.5)
meta=torch.load(run/'constructor.pt',map_location='cpu',weights_only=False);nfit=meta['fit_n']
source=OUT/'indices_escala_Train_s0';assert sha(source/'indices_fuentes.npz')==result['sourceindices_sha256']
with np.load(source/'indices_fuentes.npz') as z:
 index=z['train_index'][nfit:].copy();row=z['row'][nfit:].copy();timestamp=z['timestamp'][nfit:].copy()
 refs={key:z[key][nfit:].copy() for key in ['parent','first','last']}
saved=dict(np.load(run/'predicciones_held_Train.npz'))
assert np.array_equal(row,saved['row']) and np.array_equal(timestamp,saved['timestamp'])
d,X,names=feature_input('train',read(BEST/'resultado.json'));keep=np.array([not n.startswith('mm_') for n in names]);names=[n for n,k in zip(names,keep) if k]
X=np.ascontiguousarray(X[index][:,keep]);assert names==meta['features'] and len(X)==10000
dd={key:d[key][index] for key in ['o_lon','o_lat','d_lon','d_lat','timestamp']}
assert np.array_equal(d['row'][index],row)
bank=ArcTransferBank('cuda');tokens={mode:torch.empty((10000,64,38),device='cuda',dtype=torch.float16) for mode in ['observed','geometry']}
for a in range(0,10000,256):
 b=min(a+256,10000);queries={k:v[a:b] for k,v in dd.items()};rr={k:v[a:b] for k,v in refs.items()}
 for mode in ['observed','geometry']:
  tokens[mode][a:b]=tensor_tokens(bank,queries,X[a:b],**rr,shape_indices=meta['shape_indices'],available_indices=meta['available_indices'],geometry_proxy=mode=='geometry')
probe=np.sort(np.random.default_rng(20261128).choice(10000,65,replace=False))
items=[{key:int(value[i]) if key=='timestamp' else float(value[i]) for key,value in dd.items()} for i in probe]
rawX,rawnames=build_public206(items,read(BEST/'resultado.json')['config'])
assert rawnames==names and np.array_equal(rawX,X[probe]),float(np.max(np.abs(rawX-X[probe])))
memory=OrderedSourceMemory(source_arrival_before=meta['source_cutoff']);memory.build_tree(5)
query={key:value[probe] for key,value in dd.items()};fresh=memory.query(query)
for k,key in enumerate(['parent','first','last']):assert np.array_equal(fresh[k],refs[key][probe]),key
rawtokens={}
for mode in ['observed','geometry']:
 rawtokens[mode]=tensor_tokens(bank,query,rawX,parent=fresh[0],first=fresh[1],last=fresh[2],shape_indices=meta['shape_indices'],available_indices=meta['available_indices'],geometry_proxy=mode=='geometry')
 assert torch.equal(rawtokens[mode],tokens[mode][probe])
del memory;gc.collect()
gx=torch.tensor(X,device='cuda');rx=torch.tensor(rawX,device='cuda');checks={}
for item in result['checkpoints']:
 mode=item['arm'];path=Path(item['file']).resolve();assert path.is_relative_to(run.resolve()) and sha(path)==item['sha256']
 checkpoint=torch.load(path,map_location='cpu',weights_only=False)
 model=IndividualSourceMixture(**checkpoint['constructor']).cuda();model.load_state_dict(checkpoint['state']);model.eval()
 key='geometry' if mode=='geometry' else 'observed'
 cq=stable_canonical_quantiles(model,gx,tokens[key]).numpy();error=float(np.max(np.abs(cq-saved[mode+'_quantiles'])))
 assert error<=.001,(mode,error)
 raw=stable_canonical_quantiles(model,rx,rawtokens[key]).numpy();raw_error=float(np.max(np.abs(raw-cq[probe])));assert raw_error<=.001
 reverse=stable_canonical_quantiles(model,rx[:17].flip(0),rawtokens[key][:17].flip(0)).numpy()
 singleton=stable_canonical_quantiles(model,rx[:1],rawtokens[key][:1]).numpy()
 assert np.array_equal(raw[:17],reverse[::-1]) and np.array_equal(raw[:1],singleton)
 checks[mode]=dict(full10000_quantiles_maxerror_s=error,raw65_maxerror_s=raw_error,raw_sourceindices_and_tokens_exact=True,
 raw_reverse17_singleton_exact=True,computed_metrics=metric(saved['true'],cq[:,1]),model_sha256=item['sha256'])
 assert max(abs(checks[mode]['computed_metrics'][k]-result['metrics'][mode][k]) for k in ['mae_s','rmse_s','mdae_s'])<=.001
 print(json.dumps(dict(stage='all_data_replay_arm_complete',arm=mode,result=checks[mode],seconds=time.time()-start)),flush=True)
 del model,checkpoint;gc.collect();torch.cuda.empty_cache()
proof=dict(passed=True,scope='All10000heldquantiles and raw65queries for all3fixed4epochmodels.',fit_n=nfit,held_n=10000,arms=checks,
 raw_query=['o_lon','o_lat','d_lon','d_lat','timestamp'],source_cutoff=meta['source_cutoff'],sameheldcohort=True,
 original_labels_unchanged=True,Validation=False,Test=False,actual_data_files_accessed=sorted(access),source_sha256=sha(__file__),seconds=time.time()-start)
write(run/'reproduccion_independiente.json',proof);print(json.dumps(proof),flush=True)
