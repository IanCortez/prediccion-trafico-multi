"""Frozen GRU public rollout on all Train/Validation; never constructs a private teacher."""
from base_replanteo import *
import torch,gc,shutil
from inferencia import feature_input
from datos_progreso_recurrente import load_graph
from modelo_progreso_recurrente import ProgressNet
access=deny_test();start=time.time();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
torch.cuda.set_per_process_memory_fraction(.5)
run=OUT/'progreso_vial_recurrente_s0';stage=read(run/'resultado_etapa_A.json');assert stage['complete']
modelpath=Path(stage['model_file']);assert sha(modelpath)==stage['model_sha256']
state=torch.load(modelpath,map_location='cpu',weights_only=False)
model=ProgressNet(state['state']['mean'],state['state']['std'],load_graph())
model.load_state_dict(state['state']);model=model.cuda().eval();del state;gc.collect()
batch=1024;folder=run/'resumenes_publicos';folder.mkdir(exist_ok=True)
sources=['generar_resumenes_progreso_recurrente.py','modelo_progreso_recurrente.py','datos_progreso_recurrente.py']
manifest=dict(model_sha256=stage['model_sha256'],canonical_batch=batch,paths=16,max_steps=192,features=30,
 sources={n:sha(OUT/n) for n in sources},no_private_teacher_constructed=True,Test='sealed')
if (folder/'protocolo.json').exists():assert read(folder/'protocolo.json')==manifest
else:
 write(folder/'protocolo.json',manifest);(folder/'fuentes').mkdir()
 for name in sources:shutil.copy2(OUT/name,folder/'fuentes'/name)
progressfile=folder/'progreso.json';progress=read(progressfile) if progressfile.exists() else {}
results={}
for split in ['train','validation']:
 final=folder/(split+'_resumenes.npz');infofile=folder/(split+'_resultado.json')
 if infofile.exists():
  info=read(infofile);assert sha(final)==info['sha256'];results[split]=info;continue
 d,X,names=feature_input(split,read(BEST/'resultado.json'));n=len(X)
 xx=torch.tensor(X,device='cuda');del X;ts=torch.as_tensor(d['timestamp'],device='cuda')
 p=dict(np.load(OUT/'banco_valor_vial_s0'/(split+'_proyecciones.npz')))
 assert np.array_equal(d['row'],p['row']) and np.array_equal(d['timestamp'],p['timestamp'])
 bank={k:torch.as_tensor(p[k].astype(np.int64) if k=='chain' else p[k],device='cuda') for k in ['chain','fraction','error_m','remaining_km','reachable']};del p
 temp=folder/(split+'_avance.npy');first=int(progress.get(split,0))
 assert first==0 or (temp.exists() and first%batch==0)
 summary=np.lib.format.open_memmap(temp,mode='r+' if temp.exists() else 'w+',dtype=np.float32,shape=(n,30))
 tick=time.time()
 for a in range(first,n,batch):
  b=min(a+batch,n);count=b-a
  xq=xx[a:b];tq=ts[a:b];bb={k:v[a:b] for k,v in bank.items()}
  if count<batch:
   xq=torch.cat([xq,xq[-1:].expand(batch-count,-1)])
   tq=torch.cat([tq,tq[-1:].expand(batch-count)])
   bb={k:torch.cat([v,v[-1:].expand((batch-count,)+v.shape[1:])]) for k,v in bb.items()}
  q=model.public_rollout(xq,tq,**bb);summary[a:b]=q[:count].cpu().numpy()
  if a%(batch*16)==0:
   summary.flush();progress[split]=a+batch;write(progressfile,progress)
   report=dict(split=split,completed=b,n=n,seconds=time.time()-tick,queries_per_second=(b-first)/max(time.time()-tick,.01))
   print(json.dumps(report),flush=True)
 assert np.isfinite(summary).all()
 if split=='validation':
  root=np.load(run/'validacion_solo_raiz.npz')
  assert np.array_equal(summary[:,:3],root['quantiles'])
 np.savez_compressed(final,row=d['row'],timestamp=d['timestamp'],summary=summary)
 check=np.load(final);assert np.array_equal(check['summary'],summary)
 info=dict(n=n,sha256=sha(final),shape=list(summary.shape),bytes=final.stat().st_size,
  omitted_pair_mass_mean=float(summary[:,5].mean()),weighted_tail_mass_mean=float(summary[:,7].mean()),
  weighted_completion_mass_mean=float(summary[:,8].mean()),weighted_steps_mean=float(summary[:,6].mean()),
  style_eta_dispersion_s_mean=float(summary[:,17:21].std(1).mean()),model_sha256=stage['model_sha256'],
  no_actual_query_route_or_length_or_T=True,seconds=time.time()-tick)
 write(infofile,info);results[split]=info
 del summary,check,xx,ts,bank,d;gc.collect();torch.cuda.empty_cache()
 assert temp.resolve().is_relative_to(folder.resolve());temp.unlink()
 progress[split]=n;write(progressfile,progress)
total=sum(z['bytes'] for z in results.values());assert total<500*1024**2
write(folder/'resultado.json',dict(complete=True,outputs=results,total_MiB=total/1024**2,
 model_sha256=stage['model_sha256'],source_manifest_sha256=sha(folder/'protocolo.json'),
 actual_data_files_accessed=sorted(access),no_private_teacher_constructed=True,Test=False,seconds=time.time()-start))
print(json.dumps(dict(stage='all_public_rollouts_complete',results=results,seconds=time.time()-start)),flush=True)
