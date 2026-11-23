"""Independentcompleteheld10k sourcegraph/gate replay before finalclosure."""
from base_replanteo import *
import torch,gc
from operador_red_cinetica import RoadClockKernel,RoadStateGeometry,load_counts
from operador_red_cinetica_GPU import CudaRoadClock
from modelo_compuerta_red_cinetica import RoadCDFGate,canonical_gate,point_quantiles
from modelo_control_lognormal_puente import PublicLognormalControl,canonical_predict_control
from datos_mezcla_individual_Train import build_public206
from puente_temporal_2d import score
start=time.time();access=deny_test();attempts=[]
forbidden=(OUT/'memoria_puntos_Train_s0/puntos_Train.npz').resolve()
def no_points(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
  try:p=Path(os.fsdecode(args[0])).resolve()
  except (OSError,ValueError):return
  if p==forbidden:attempts.append(str(p));raise AssertionError('Replaycannot open originalpointbank')
sys.addaudithook(no_points)
torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.5)
run=OUT/'red_cinetica_Train_s0';r=read(run/'resultado.json');assert r['complete']
for name,digest in r['sources'].items():assert sha(run/'fuentes'/name)==digest and sha(OUT/name)==digest
for file,key in [('protocolo.json','protocol_sha256'),('constructor.pt','constructor_sha256'),('predicciones_held_Train.npz','predictions_sha256')]:assert sha(run/file)==r[key]
inputs=OUT/'indices_red_cinetica_Train_s0';datafile=inputs/'consultas_predicciones.npz';assert sha(datafile)==r['input_sha256']
with np.load(datafile) as z:
 nfit=int(z['fit_n']);index=z['train_index'][nfit:];row=z['row'][nfit:];timestamp=z['timestamp'][nfit:];X=z['publicX'][nfit:]
 refprob=z['reference_probability'][nfit:]
 cached_graph={kind:z[kind+'_probability'][nfit:] for kind in ['position','kinetic']}
 cached_supported={kind:z[kind+'_supported'][nfit:] for kind in cached_graph}
saved=dict(np.load(run/'predicciones_held_Train.npz'))
assert np.array_equal(saved['row'],row) and np.array_equal(saved['timestamp'],timestamp)
with np.load(DATA/'train_x.npz') as z:
 assert np.array_equal(z['row'][index],row) and np.array_equal(z['timestamp'][index],timestamp)
 query={key:z[key][index] for key in ['o_lon','o_lat','d_lon','d_lat','timestamp']}
probe=np.sort(np.random.default_rng(20261204).choice(10000,65,replace=False))
items=[{key:int(value[i]) if key=='timestamp' else float(value[i]) for key,value in query.items()} for i in probe]
raw,names=build_public206(items,read(BEST/'resultado.json')['config']);assert np.array_equal(raw,X[probe]) and len(names)==206
reference=read(OUT/'referencia_publica_Train_red_vial.json');assert sha(Path(reference['model_file']))==reference['model_sha256']
ck=torch.load(reference['model_file'],map_location='cpu',weights_only=False);model=PublicLognormalControl(**ck['constructor']).cuda();model.load_state_dict(ck['state'])
x=torch.as_tensor(X,device='cuda');lp0,q0=canonical_predict_control(model,x);assert np.array_equal(lp0.exp().numpy(),refprob)
assert np.array_equal(q0[:,1].numpy(),saved['reference']);logref=lp0.cuda();del model,ck;gc.collect();torch.cuda.empty_cache()
countfile=OUT/'conteos_cineticos_prefijo_Train_s0/conteos_fuentes.npz';assert sha(countfile)==r['source_count_sha256']
counts=load_counts(countfile);geometry=RoadStateGeometry(DATA/'mapa_dirigido_2014.npz');proofs={}
for kind in ['position','kinetic']:
 phase=time.time();cp=read(run/kind/'resultado.json');data=dict(np.load(run/kind/'predicciones_held_Train.npz'))
 assert sha(run/kind/'predicciones_held_Train.npz')==cp['predictions_sha256']
 checkpoint=Path(r['weights'][kind]['file']);assert sha(checkpoint)==r['weights'][kind]['sha256']
 state=torch.load(checkpoint,map_location='cpu',weights_only=False);gate=RoadCDFGate(**state['constructor']).cuda();gate.load_state_dict(state['state']);gate.eval()
 supported=torch.as_tensor(cached_supported[kind],device='cuda');graph=torch.as_tensor(cached_graph[kind],device='cuda',dtype=torch.float64)
 graph=torch.where(supported[:,None],graph/graph.sum(1).clamp_min(1e-300)[:,None],torch.zeros_like(graph))
 lp,g=canonical_gate(gate,x,logref,graph,supported);q=point_quantiles(lp)
 assert np.array_equal(lp.numpy(),data['log_prob']) and np.array_equal(g.numpy(),data['gate'])
 assert np.array_equal(q.numpy(),data['quantiles']) and np.array_equal(q[:,1].numpy(),saved[kind])
 assert metric(saved['true'],saved[kind])==r['metrics'][kind]
 _,crps,nll=score(lp,torch.as_tensor((saved['true']/15).astype(np.int64)))
 assert abs(float(crps.mean())-r['probability_scores'][kind]['CRPS_s'])<1e-10
 assert abs(float(nll.mean())-r['probability_scores'][kind]['bin_NLL'])<1e-10
 reverse,rg=canonical_gate(gate,x[probe[:17]].flip(0),logref[probe[:17]].flip(0),graph[probe[:17]].flip(0),supported[probe[:17]].flip(0))
 single,sg=canonical_gate(gate,x[probe[:1]],logref[probe[:1]],graph[probe[:1]],supported[probe[:1]])
 assert torch.equal(reverse.flip(0),lp[probe[:17]]) and torch.equal(single,lp[probe[:1]])
 assert torch.equal(rg.flip(0),g[probe[:17]]) and torch.equal(sg,g[probe[:1]])
 # Recompute all10000 endpointgraphdistributions in a freshsourcekernel, using raw5 only.
 torch.use_deterministic_algorithms(True);cpu=RoadClockKernel(counts,geometry,kind=kind,excluded_fold=5);operator=CudaRoadClock(cpu)
 last=[0.]
 def progress(detail):
  now=time.time()
  if now-last[0]>30:
   last[0]=now;print(json.dumps(dict(stage='independent_graph_replay',kind=kind,**detail,seconds=now-start)),flush=True)
 generated=operator.predict(query,progress);assert np.array_equal(generated['supported'],cached_supported[kind])
 graph32=generated['probability'].astype(np.float32)
 difference=float(np.max(np.abs(graph32-cached_graph[kind]),initial=0));exact=bool(np.array_equal(graph32,cached_graph[kind]))
 assert difference<1e-7
 check=cpu.predict({key:value[probe[:3]] for key,value in query.items()})
 assert np.max(np.abs(check['probability']-generated['probability'][probe[:3]]),initial=0)<1e-10
 reversed_graph=operator.predict({key:value[probe[:17][::-1]] for key,value in query.items()})
 singleton_graph=operator.predict({key:value[probe[:1]] for key,value in query.items()})
 assert np.max(np.abs(reversed_graph['probability'][::-1]-generated['probability'][probe[:17]]),initial=0)<1e-10
 assert np.max(np.abs(singleton_graph['probability']-generated['probability'][probe[:1]]),initial=0)<1e-10
 torch.use_deterministic_algorithms(False)
 generated_graph=torch.as_tensor(graph32,device='cuda',dtype=torch.float64)
 generated_graph=torch.where(supported[:,None],generated_graph/generated_graph.sum(1).clamp_min(1e-300)[:,None],torch.zeros_like(generated_graph))
 replay_lp,replay_g=canonical_gate(gate,x,logref,generated_graph,supported);replay_q=point_quantiles(replay_lp)
 assert torch.equal(replay_q,q),'Rebuiltall10000graphs must yield identicalpointquantiles'
 # Probabilityagreement measured explicitly; pointforecast equality is mandatory.
 pdiff=float(np.max(np.abs(replay_lp.exp().numpy()-np.exp(data['log_prob'])),initial=0));assert pdiff<1e-7
 history=read(run/kind/'historia.json');assert len(history)==80 and sum(item['updates'] for item in history)==12560
 proofs[kind]=dict(passed=True,all10000_cached_logprob_andgates_Float64_exact=True,all10000_quantiles_exact=True,
  raw65public206_exact=True,gate_reverse17_singleton_exact=True,all10000_raw5_graph_support_exact=True,
  all10000_raw5_graph_probability_FP32_exact=exact,all10000_raw5_graph_max_probability_error=difference,
  all10000_raw5_endtoend_quantiles_exact=True,endtoend_max_probability_error=pdiff,
  sparse_GPU_CPU_first3_andraw_reverse17_singleton_within1e_10=True,fixed12560updates_verified=True,
  computed_metrics=r['metrics'][kind],seconds=time.time()-phase)
 print(json.dumps(dict(stage='independent_arm_complete',kind=kind,proof=proofs[kind])),flush=True)
 del gate,state,graph,generated_graph,supported,cpu,operator,generated,lp,g,replay_lp;gc.collect();torch.cuda.empty_cache()
for name,a,b in [('position_vs_reference','position','reference'),('kinetic_vs_reference','kinetic','reference'),('kinetic_vs_position','kinetic','position')]:
 pair=paired(saved['true'],saved[a],saved[b],saved['timestamp']);expected=r['comparisons'][name]
 assert abs(pair['delta_mae_s']-expected['delta_mae_s'])<1e-12 and np.array_equal(pair['ci95_s'],expected['ci95_s'])
report=dict(passed=True,scope='Independentcompleteall10000raw5graph and bothgate replay; no originalpointbank.',arms=proofs,
 all_comparisons_andday_bootstrap_recomputed=True,reference_all10000_exact=True,original_pointbank_open_attempts=attempts,
 queryfields=list(query),query_N_T_actualroadIDs_or_path_supplied=False,
 actual_data_files_accessed=sorted(access),seconds=time.time()-start,source_sha256=sha(__file__),Validation=False,Test=False)
write(run/'reproduccion_independiente.json',report);print((run/'reproduccion_independiente.json').read_text(encoding='utf-8'),flush=True)
