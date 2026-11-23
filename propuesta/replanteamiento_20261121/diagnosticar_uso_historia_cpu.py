"""Label-blind bounded CPU probe of a stable live-training checkpoint."""
from base_replanteo import *
import torch
from inferencia import feature_input
from modelo_mezcla_rutas import make_duration_model
from modelo_mezcla_proyeccion import ProjectionMixture
from modelo_mezcla_historica import HistoricalRouteMixture
access=deny_test();start=time.time();torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
run=OUT/'historica_fresca_s0';path=run/'ultima_reanudacion.pt'
before=path.stat();state=torch.load(path,map_location='cpu',weights_only=False);snapshot_sha=sha(path);after=path.stat()
assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns,'Checkpoint changed during read; inspect/retry same run, never restart training.'
epoch=int(state['epoch']);target=run/('diagnostico_uso_historia_epoca_'+str(epoch)+'.json');assert not target.exists()
ctor=torch.load(run/'constructor_base.pt',map_location='cpu',weights_only=False);hc=dict(np.load(run/'constructor_historia.npz'));mc=dict(np.load(run/'constructor_mapa.npz'))
rec=read(BEST/'resultado.json');duration=make_duration_model(ctor,rec['config']);parent=ProjectionMixture(duration,mc['mean'],mc['std'],'expanded')
model=HistoricalRouteMixture(parent,hc['mean'],hc['std'],hc['time_anchor']).eval();model.load_state_dict(state['state'])
d,V,names=feature_input('validation',rec);ii=np.sort(np.random.default_rng(1729).choice(len(V),65,replace=False))
C=np.load(OUT/'banco_proyeccion_incierta_s0/validation_banco.npz')['bank'][ii]
H=np.load(OUT/'datos_prototipos_historicos_s0/validation_historicos.npz')['history'][ii]
x=torch.from_numpy(V[ii]);c=torch.from_numpy(C);h=torch.from_numpy(H)
with torch.no_grad():
 out=model(x,c,h);base=model.base_only(x,c)
 q=model.duration.quantiles(out[0]);qb=model.duration.quantiles(base[0])
 mass=out[1].exp().mean(1);general=mass[:,:8].sum(-1).numpy();public=mass[:,8:16].sum(-1).numpy();history=mass[:,16:].sum(-1).numpy()
 shift=float((q[:,1]-qb[:,1]).abs().mean())
 delta=(out[2][:,:,16:,:]-h.float()[:,None,:,:14]).reshape(65,16,6,7,2).norm(dim=-1).mean()
result=dict(epoch=epoch,n=65,scope='Input-only usage probe, no accuracy/MAE conclusion.',snapshot_sha256=snapshot_sha,checkpoint_read_stable=True,device='CPU2threads',
 mean_mass_general=float(general.mean()),mean_mass_public=float(public.mean()),mean_mass_historical=float(history.mean()),
 historical_mass_quantiles=np.quantile(history,[.1,.5,.9]),mean_absolute_median_shift_due_to_historical_branch_s=shift,
 mean_learned_geometry_shift_from_historical_prototype_m=float(delta)*1000,
 mean_historical_geo_sigma_km=float(out[3][:,:,16:,:].mean()),
 historical_inputs_available_fraction=float(np.mean((H[:,:,23]>.5).any(1))),source_sha256=sha(__file__),
 Validation_labels_accessed=False,Test_access=False,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
assert not any(Path(p).name=='validation_y.npy' for p in access)
write(target,result);print(json.dumps({k:result[k] for k in ['epoch','n','mean_mass_general','mean_mass_public','mean_mass_historical','mean_absolute_median_shift_due_to_historical_branch_s','seconds']}),flush=True)
