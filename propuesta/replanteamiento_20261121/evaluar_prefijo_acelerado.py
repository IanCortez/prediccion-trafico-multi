"""Inference-only replacement after training finished. Batch choice never uses MAE."""
from base_replanteo import *
from modelo_prefijo_nativo import NativePrefixArrival
from datos_prefijo_nativo import query_seeds
from modelo_mezcla_rutas import make_duration_model
from inferencia import feature_input
import torch,shutil
torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
torch.cuda.set_per_process_memory_fraction(.35,0)
access=deny_test()
start=time.time()
run=OUT/'prefijo_nativo_s0'
assert not (run/'resultado.json').exists()
freeze=read(run/'congelacion_para_inferencia.json')
assert sha(run/'modelo.pt')==freeze['model_sha256']
assert freeze['epochs_completed']==10 and freeze['steps']==79960
for name in ['modelo_prefijo_nativo.py','datos_prefijo_nativo.py','modelo_dir_fases.py','modelo_dir_prior.py','modelo_direccional.py','modelo_espacial.py']:
    assert sha(OUT/name)==sha(run/'fuentes'/name)
audit=run/'evaluacion_acelerada'
audit.mkdir(exist_ok=True)
sources=audit/'fuentes';sources.mkdir(exist_ok=True)
shutil.copy2(OUT/'evaluar_prefijo_acelerado.py',sources/'evaluar_prefijo_acelerado.py')
ck=torch.load(run/'modelo.pt',map_location='cpu',weights_only=False)
graph=dict(np.load(OUT/'datos_semi_markov/grafo_grid.npz'))
distance=np.load(OUT/'datos_semi_markov/distancias_grid.npy')
counts=np.load(OUT/'prior_transiciones/conteos_fold.npz')['counts']
model=NativePrefixArrival(ck['stats'],ck['grid'],graph,distance,ck['power'],counts,ck['phase_initialization'])
model.load_state_dict(ck['state'])
for name in ['counts_fold','counts_base_fold','counts_global_fold']:setattr(model,name,torch.empty(0))
model=model.cuda().eval()
grid=ck['grid'];parent_sha=ck['parent_sha256']
del ck,counts
rec=read(BEST/'resultado.json')
va,V,names=feature_input('validation',rec)
assert names==rec['features']
vx=torch.tensor(V,device='cuda');del V
old=np.load(BEST/'validacion.npz')
y=old['true'];reference_pred=old['pred']
assert np.array_equal(old['row'],va['row'])
def ids(tag):
    xx=(va[tag+'_lon']+8.62)*111195*np.cos(np.deg2rad(41.16))
    yy=(va[tag+'_lat']-41.16)*111195
    return torch.tensor(np.floor((xx-grid['x0'])/200).astype(np.int64)*grid['ny']+np.floor((yy-grid['y0'])/200).astype(np.int64),device='cuda')
origin,destination=ids('o'),ids('d')
seeds=torch.tensor(query_seeds(va),device='cuda')
original=torch.load(BEST/'modelo.pt',map_location='cpu',weights_only=False)
teacher=make_duration_model(original,rec['config']).cuda().eval();teacher.load_state_dict(original['state'])
reference=torch.empty((len(y),116),device='cuda')
with torch.no_grad():
    for first in range(0,len(y),2048):
        block=vx[first:first+2048];n=len(block)
        if n<2048:block=torch.cat([block,block[-1:].expand(2048-n,-1)])
        reference[first:first+n]=teacher(block)[0].exp().mean(1)[:n]
assert np.array_equal(model.median(reference).cpu().numpy(),reference_pred)
del teacher,original
torch.cuda.empty_cache()
cfg=read(run/'protocolo.json')
alphas=cfg['alphas']
@torch.no_grad()
def block_predict(indices,batch):
    ix=torch.tensor(indices,device='cuda')
    n=len(ix)
    assert 0<n<=batch
    if n<batch:ix=torch.cat([ix,ix[-1:].expand(batch-n)])
    raw,left=model.generated_arrival(vx[ix],origin[ix],destination[ix],seeds[ix],particles=128)
    assert torch.isfinite(raw).all() and (raw>=0).all()
    assert ((raw.sum(-1)+left-1).abs()<2e-4).all()
    mass=raw[:,4:120].sum(-1)
    path=raw[:,4:120]/mass[:,None].clamp_min(1e-12)
    support=mass/(mass+.01)
    out=[]
    for alpha in alphas:
        if alpha==0:p=reference[ix]
        else:
            a=alpha*support
            p=(1-a[:,None])*reference[ix]+a[:,None]*path
            p=p/p.sum(-1,keepdim=True)
        out.append(model.median(p)[:n].cpu().numpy())
    return np.stack(out),mass[:n].cpu().numpy()
def predict(indices,batch):
    pp=[];mm=[]
    for first in range(0,len(indices),batch):
        p,m=block_predict(indices[first:first+batch],batch)
        pp.append(p);mm.append(m)
    return np.concatenate(pp,axis=1),np.concatenate(mm)
probe=np.load(run/'selection_indices.npy')
torch.cuda.synchronize();t=time.time()
base,base_mass=predict(probe,16)
torch.cuda.synchronize()
bench=[dict(batch=16,seconds=time.time()-t,point_changes=0,mass_max_error=0.,eligible=True)]
selected_batch=16;best_seconds=bench[0]['seconds']
for batch in [64,128]:
    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize();t=time.time()
        p,m=predict(probe,batch)
        torch.cuda.synchronize()
        seconds=time.time()-t
        changes=int((p!=base).sum());error=float(np.max(np.abs(m-base_mass)))
        eligible=changes==0 and error<2e-5
        bench.append(dict(batch=batch,seconds=seconds,point_changes=changes,mass_max_error=error,peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,eligible=eligible))
        if eligible and seconds<best_seconds:selected_batch=batch;best_seconds=seconds
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        bench.append(dict(batch=batch,eligible=False,reason='Recoverable allocator OOM; smaller verified batch retained'))
if (audit/'progreso_predicciones.npz').exists():
    selected_batch=read(audit/'protocolo.json')['config']['canonical_query_batch']
    assert any(r['batch']==selected_batch and r['eligible'] for r in bench)
write(audit/'seleccion_lote.json',dict(rule='Fastest eligible batch among16,64,128; require exact5-output point equality on fixed512 and mass error<2e-5; never inspect MAE for this choice',measurements=bench,chosen=selected_batch,scope='Probe equivalence only, not a claim of universal equality with unfinished legacy inference'))
# Verify canonical prediction does not depend on query order or neighboring queries.
check=probe[:65]
p,_=predict(check,selected_batch)
reverse,_=predict(check[::-1].copy(),selected_batch)
assert np.array_equal(p,reverse[:,::-1])
for j in [0,19,42,64]:
    one,_=predict(check[j:j+1],selected_batch)
    assert np.array_equal(one[:,0],p[:,j])
write(audit/'verificacion_inferencia.json',dict(passed=True,reference_all127941_exact=True,probe_queries=512,query_permutation65=True,singleton4=True,chosen_batch=selected_batch,particles=128,weights_unchanged=sha(run/'modelo.pt')==freeze['model_sha256'],no_Test=True))
print(json.dumps(dict(stage='accelerated_inference_ready',batch=selected_batch,bench=bench)),flush=True)
selection,_=predict(probe,selected_batch)
selected=int(np.argmin([metric(y[probe],p)['mae_s'] for p in selection]))
assert alphas[selected]==read(run/'estabilidad_monte_carlo.json')['selected_alpha']
cfg=dict(cfg,canonical_query_batch=selected_batch,inference_precision='FP32 highest',inference_only_replacement=True)
write(audit/'protocolo.json',dict(config=cfg,training_weights_sha256=freeze['model_sha256'],training_not_repeated=True,reference_batch=2048,reason='Verified training finished; original canonical16 inference had low throughput. Use fastest probe-identical batch chosen without MAE.',Test_access=False))
pred=np.full((len(alphas),len(y)),np.nan,np.float32);mass=np.full(len(y),np.nan,np.float32)
checkpoint=audit/'progreso_predicciones.npz'
completed=0
if checkpoint.exists():
    saved=np.load(checkpoint)
    assert int(saved['batch'])==selected_batch
    completed=int(saved['completed'])
    pred[:,:completed]=saved['pred'];mass[:completed]=saved['mass']
last_save=completed
for first in range(completed,len(y),selected_batch):
    last=min(first+selected_batch,len(y))
    pp,mm=block_predict(np.arange(first,last),selected_batch)
    pred[:,first:last]=pp;mass[first:last]=mm
    if last-last_save>=8192 or last==len(y):
        temp=audit/'progreso_temporal.npz'
        np.savez_compressed(temp,pred=pred[:,:last],mass=mass[:last],completed=last,batch=selected_batch)
        os.replace(temp,checkpoint);last_save=last
        print(json.dumps(dict(stage='accelerated_full_validation',completed_rows=last,total_rows=len(y),seconds=time.time()-start)),flush=True)
assert np.isfinite(pred).all() and np.array_equal(pred[0],reference_pred)
all_metrics=[metric(y,p) for p in pred]
np.savez_compressed(run/'validacion.npz',row=va['row'],timestamp=va['timestamp'],true=y,pred=pred,alphas=np.array(alphas),selected_alpha=alphas[selected],absorption_mass=mass)
qualified=[dict(alpha=a,validation=m,robustness_512_and_reproduction_pending=True) for a,m in zip(alphas,all_metrics) if a>0 and m['mae_s']<131.06]
result=dict(id='prefijo_nativo_s0',config=cfg,training_protocol=read(run/'protocolo.json'),validation=all_metrics[selected],graph_validation=all_metrics[-1],all_alpha_validation=all_metrics,selected_alpha=alphas[selected],reference_validation=all_metrics[0],best_epoch=freeze['best_epoch'],epochs_run=10,steps=79960,model_sha256=sha(run/'modelo.pt'),code_sha256=sha(run/'fuentes/entrenar_prefijo_nativo.py'),inference_code_sha256=sha(__file__),model_code_sha256=sha(OUT/'modelo_prefijo_nativo.py'),data_code_sha256=sha(OUT/'datos_prefijo_nativo.py'),parent_sha256=parent_sha,reference_sha256=rec['model_sha256'],paired_vs_retained=paired(y,pred[selected],reference_pred,va['timestamp']),qualified_candidates=qualified,new_primary_best_pending_reproduction=all_metrics[selected]['mae_s']<all_metrics[0]['mae_s'],goal_validation=all_metrics[selected]['mae_s']<73.5,actual_data_files_accessed_in_replacement_inference=sorted(access),training_trace='Unchanged completed checkpoint, final resume10epochs/79960steps, immutable training sources and history; original process stopped during slow inference, not training',training_elapsed_seconds=freeze['training_elapsed_seconds'],replacement_inference_seconds=time.time()-start,Test_access=False,inference_audit=str(audit))
write(run/'resultado.json',result)
saved=np.load(run/'validacion.npz')
assert np.array_equal(saved['pred'],pred)
if checkpoint.exists():
    assert checkpoint.resolve().is_relative_to(ROOT)
    checkpoint.unlink()
write(audit/'limpieza.json',dict(temporary_prediction_checkpoint_removed=True,final_predictions_verified=True,no_distribution_bank_on_disk=True))
print(json.dumps({k:result[k] for k in ['validation','graph_validation','all_alpha_validation','qualified_candidates','replacement_inference_seconds']}),flush=True)
