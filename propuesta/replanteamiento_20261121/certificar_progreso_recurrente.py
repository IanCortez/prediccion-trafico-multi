"""Independent frozen-candidate replay: full public rollout plus raw65, no Test."""
from base_replanteo import *
import torch,importlib.util,argparse,shutil,gc
from inferencia import feature_input
from inferencia_valor_vial import raw_inputs
from catboost import CatBoostRegressor
ap=argparse.ArgumentParser();ap.add_argument('--name',default='full246_MAE');args=ap.parse_args()
access=deny_test();start=time.time();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
torch.cuda.set_per_process_memory_fraction(.5)
run=OUT/'progreso_vial_recurrente_s0';bf=run/'etapa_B';result=read(bf/'resultado.json')
info=result['models'][args.name];assert info['preserved']
tree=CatBoostRegressor();assert sha(bf/info['file'])==info['sha256'];tree.load_model(str(bf/info['file']))
artifact=run/('certificacion_'+args.name);assert not artifact.exists();artifact.mkdir()
for n in ['certificar_progreso_recurrente.py','inferencia_valor_vial.py','proyecciones_valor_vial.py']:
 shutil.copy2(OUT/n,artifact/n)
full=args.name.startswith('full246');model=None
if full:
 stage=read(run/'resultado_etapa_A.json');assert sha(stage['model_file'])==stage['model_sha256']
 for n,value in stage['sources'].items():assert sha(run/'fuentes'/n)==value
 spec=importlib.util.spec_from_file_location('frozen_recurrent_progress',run/'fuentes/modelo_progreso_recurrente.py')
 module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 data_spec=importlib.util.spec_from_file_location('frozen_progress_public_graph',run/'fuentes/datos_progreso_recurrente.py')
 dm=importlib.util.module_from_spec(data_spec);sys.modules[data_spec.name]=dm;data_spec.loader.exec_module(dm)
 state=torch.load(stage['model_file'],map_location='cpu',weights_only=False)
 model=module.ProgressNet(state['state']['mean'],state['state']['std'],dm.load_graph())
 model.load_state_dict(state['state']);model=model.cuda().eval();del state
batch=1024
def summaries(X,timestamp,bank):
 parts=[]
 with torch.no_grad():
  for first in range(0,len(X),batch):
   end=min(first+batch,len(X));n=end-first
   xx=torch.as_tensor(X[first:end],device='cuda')
   ts=torch.as_tensor(timestamp[first:end],device='cuda')
   b={k:torch.as_tensor(v[first:end].astype(np.int64) if k=='chain' else v[first:end],device='cuda') for k,v in bank.items()}
   if n<batch:
    xx=torch.cat([xx,xx[-1:].expand(batch-n,-1)]);ts=torch.cat([ts,ts[-1:].expand(batch-n)])
    b={k:torch.cat([v,v[-1:].expand((batch-n,)+v.shape[1:])]) for k,v in b.items()}
   parts.append(model.public_rollout(xx,ts,**b)[:n].cpu().numpy())
 return np.concatenate(parts)
def point(X,summary=None):
 if full:X=np.column_stack([X,summary])
 return (tree.predict(X,thread_count=2)*600).clip(60,1785)
d,V,names=feature_input('validation',read(BEST/'resultado.json'));z=dict(np.load(bf/'validacion.npz'))
truth=np.load(DATA/'validation_y.npy');assert len(truth)==127941 and np.array_equal(z['true'],truth)
for k in ['row','timestamp']:assert np.array_equal(z[k],d[k])
bankdata=dict(np.load(OUT/'banco_valor_vial_s0/validation_proyecciones.npz'))
bank={k:bankdata[k] for k in ['chain','fraction','error_m','remaining_km','reachable']}
summary=summaries(V,d['timestamp'],bank) if full else None
if full:
 stored=np.load(run/'resumenes_publicos/validation_resumenes.npz')['summary']
 assert np.array_equal(summary,stored),'Publicsummary fullreplaychanged'
pred=point(V,summary);assert np.array_equal(pred,z[args.name]) and metric(truth,pred)==result['metrics'][args.name]
ii=np.sort(np.random.default_rng(20261122).choice(len(V),65,replace=False))
fields=['o_lon','o_lat','d_lon','d_lat','timestamp']
items=[{k:int(d[k][i]) if k=='timestamp' else float(d[k][i]) for k in fields} for i in ii]
X,b,raw_names=raw_inputs(items);assert names==raw_names and np.max(np.abs(X-V[ii]))<1e-5
for k in bank:assert np.array_equal(b[k],bank[k][ii])
tt=np.asarray([it['timestamp'] for it in items],np.int64)
rs=summaries(X,tt,b) if full else None
rp=point(X,rs);assert np.array_equal(rp,pred[ii])
RX,RB,_=raw_inputs(items[::-1]);rr=summaries(RX,tt[::-1],RB) if full else None
SX,SB,_=raw_inputs(items[:1]);ss=summaries(SX,tt[:1],SB) if full else None
assert np.array_equal(point(RX,rr)[::-1],rp) and np.array_equal(point(SX,ss),rp[:1])
for forbidden in ['route','N','duration','driver','call','length']:
 try:raw_inputs([dict(items[0],**{forbidden:777})])
 except AssertionError:pass
 else:raise AssertionError(forbidden+'accepted')
write(artifact/'consultas_raw_sin_etiquetas.json',items)
write(artifact/'resultado.json',dict(passed=True,name=args.name,validation=result['metrics'][args.name],
 full127941_point_replay_exact=True,full_public_summary_replay_exact=bool(full),
 raw65_point_replay_exact=True,reverse_singleton_exact=True,extra_private_fields_rejected=True,
 tree_model_sha256=info['sha256'],sequence_model_sha256=result['sequence_model_sha256'] if full else None,
 sources={n:sha(artifact/n) for n in ['certificar_progreso_recurrente.py','inferencia_valor_vial.py','proyecciones_valor_vial.py']},
 Test_access=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start))
print(json.dumps(dict(passed=True,name=args.name,validation=result['metrics'][args.name],seconds=time.time()-start)),flush=True)
