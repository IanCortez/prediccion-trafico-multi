from common import *
from features import *
from inferencia import feature_input
from modelo_mezcla_rutas import RouteMixtureTabM
import torch,argparse,random,rtdl_num_embeddings as rtdl
p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--aux',type=float,default=.2);p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch',type=int,default=512);p.add_argument('--k',type=int,default=16);p.add_argument('--width',type=int,default=512);p.add_argument('--verify-only',action='store_true');p.add_argument('--weekly-only',action='store_true')
p.add_argument('--feature-set',choices=['legacy','projected'],default='legacy');p.add_argument('--calendar',action='store_true')
a=p.parse_args();cfg=vars(a).copy();cfg.update(routes=True,context=True,gps_routes=True,map_geometry=True,dynamic_routes=True,architecture='tabm_route_mixture')
if a.feature_set=='projected':
    assert a.aux==0,'Projected priors require the validated no-auxiliary variant'
    cfg.update(gps_routes=False,map_geometry=False,dynamic_routes=False,projected_routes=True,mapmatched=True)
started=time.time();torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed);torch.set_float32_matmul_precision('high')
tr,X,names=feature_input('train',dict(config=cfg,architecture=cfg['architecture']));va,V,_=feature_input('validation',dict(config=cfg,architecture=cfg['architecture']));y=np.load(DATA/'train_y.npy');v=np.load(DATA/'validation_y.npy')
keep=np.flatnonzero(np.ptp(X,axis=0)>1e-7);mean=X[:,keep].mean(0,dtype=np.float64).astype(np.float32);std=np.maximum(X[:,keep].std(0,dtype=np.float64),1e-5).astype(np.float32);normalized=(X[:,keep]-mean)/std
ix=np.random.default_rng(a.seed).choice(len(y),min(200000,len(y)),replace=False);sample=np.vstack([normalized[ix],normalized.min(0),normalized.max(0)]);bins=rtdl.compute_bins(torch.as_tensor(sample),n_bins=48)
anchor_names=('mm_length_gps_time_600s','mm_free_gps_time_600s','projected_free_freeflow_600s','od_km') if a.feature_set=='projected' else ('length_gps_time_600s','free_gps_time_600s','dynamic_gps_time_600s','od_km')
anchor_indices=[names.index(s) for s in anchor_names];anchors=[]
for col,scale in zip(anchor_indices,(600,600,600,1000)):
    z=np.log(np.maximum(X[:,col].astype(float)*scale,.1));beta,intercept=np.polyfit(z,np.log(y),1);intercept+=np.log(np.median(y/np.exp(intercept+beta*z)));anchors.append([intercept,beta])
model=RouteMixtureTabM(mean,std,keep,bins,anchors,anchor_indices,a.k,a.width).cuda()
def loss_fn(logp,weights,target,aux_target=None,aux_mask=None):
    label=((target-60)/15).long();nll=-logp.gather(-1,label[:,None,None].expand(-1,a.k,1)).mean()
    desired=(model.duration_grid[None,None,:]>=target[:,None,None]).float();rps=(logp.exp().cumsum(-1)-desired).square().mean()
    loss=nll+5*rps
    if a.aux and aux_target is not None:
        aux=-(weights*aux_target[:,None,:]).sum(-1).mean(1);loss=loss+a.aux*(aux*aux_mask).sum()/aux_mask.sum().clamp_min(1)
    return loss
if a.verify_only:
    inputs=torch.tensor(X[:128],device='cuda');target=torch.tensor(y[:128],device='cuda');model.eval()
    with torch.autocast('cuda',dtype=torch.bfloat16):logp,w=model(inputs);loss=loss_fn(logp,w,target)
    assert torch.isfinite(loss);assert (logp.exp().sum(-1)-1).abs().max()<1e-5
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    with torch.no_grad():
        q=model.quantiles(logp);rev,_=model(inputs.flip(0));q2=model.quantiles(rev).flip(0)
    # Different mixed-precision execution may shift probabilities at a quantile boundary.
    assert torch.all(q[:,0]<=q[:,1]) and torch.all(q[:,1]<=q[:,2]);assert torch.isfinite(q).all()
    with torch.no_grad():
        with torch.autocast('cuda',dtype=torch.bfloat16):rev,_=model(inputs.flip(0))
    assert torch.equal(q,model.quantiles(rev).flip(0))
    write_json(BASE/('instrumentos/mezcla_rutas_contrato'+('_projected' if a.feature_set=='projected' else '')+'.json'),dict(passed=True,loss=float(loss),max_probability_sum_error=float((logp.exp().sum(-1)-1).abs().max()),finite_gradients=True,permutation_invariant=True,ordered_quantiles=True,observed_query_path_input=False,test_labels_read=False))
    print('Mixture distribution and gradient contracts passed',flush=True);sys.exit(0)
run=RUNS/a.id;run.mkdir(exist_ok=False);snapshot(run,cfg)
if a.aux:
    gt=dict(np.load(DATA/'geometria_train.npz'));pos=np.searchsorted(gt['row'],tr['row']);safe=np.minimum(pos,len(gt['row'])-1);valid=(pos<len(gt['row']))&(gt['row'][safe]==tr['row'])
    shape=gt['X'][safe,1:15].reshape(-1,7,2);score=[]
    for prefix,available in (('length','length_shape_available'),('free','free_shape_available'),('dynamic','dynamic_shape_available')):
        columns=[names.index(prefix+'_shape_'+str(i)) for i in range(14)];predicted=X[:,columns].reshape(-1,7,2);error=np.linalg.norm(shape-predicted,axis=-1).mean(1)
        score.append(np.exp(-.5*(error/.2)**2)*X[:,names.index(available)])
    score=np.column_stack(score);soft=np.column_stack([score,.15+np.maximum(1-score.max(1),0)]);soft/=soft.sum(1,keepdims=True)
else:
    soft=np.zeros((len(y),4),np.float32);soft[:,-1]=1;valid=np.ones(len(y),bool)
x=torch.tensor(X,device='cuda');vx=torch.tensor(V,device='cuda');target=torch.tensor(y,device='cuda');aux=torch.tensor(soft,dtype=torch.float32,device='cuda');mask=torch.tensor(valid,dtype=torch.float32,device='cuda')
opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0003);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=5,min_lr=1e-5)
@torch.no_grad()
def predict():
    model.eval();out=[]
    for block in vx.split(2048):
        with torch.autocast('cuda',dtype=torch.bfloat16):logp,_=model(block)
        out.append(model.quantiles(logp).cpu().numpy())
    return np.concatenate(out)
best=float('inf');bad=0;history=[]
for epoch in range(1,a.epochs+1):
    model.train();total=0
    for ix in torch.randperm(len(x),device='cuda').split(a.batch):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):logp,w=model(x[ix]);loss=loss_fn(logp,w,target[ix],aux[ix],mask[ix])
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite mixture loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=loss.item()*len(ix)
    quantiles=predict();vm=metrics(v,quantiles[:,1]);scheduler.step(vm['mae_s']);rec=dict(epoch=epoch,train_loss=total/len(x),validation=vm,coverage80=float(np.mean((v>=quantiles[:,0])&(v<=quantiles[:,2]))),width80_s=float(np.mean(quantiles[:,2]-quantiles[:,0])),seconds=time.time()-started);history.append(rec);print(json.dumps(rec),flush=True)
    if vm['mae_s']<best-.02:
        best=vm['mae_s'];bad=0
        torch.save(dict(state=model.state_dict(),mean=mean,std=std,keep=keep,bins=[b.cpu() for b in bins],anchors=anchors,anchor_indices=anchor_indices,config=cfg,features=names),run/'modelo.pt')
        np.savez_compressed(run/'validacion.npz',row=va['row'],timestamp=va['timestamp'],pred=quantiles[:,1],true=v,quantiles=quantiles);write_json(run/'mejor.json',rec)
    else:bad+=1
    write_json(run/'historia.json',history)
    if bad>=20:break
z=np.load(run/'validacion.npz');q=z['quantiles'];pred=z['pred']
rec=dict(id=a.id,architecture=cfg['architecture'],config=cfg,features=names,anchor_features=anchor_names,validation=metrics(v,pred),interval=block_interval(v,pred,va['timestamp']),train_n=len(y),epochs=len(history),parameters=sum(w.numel() for w in model.parameters()),elapsed_seconds=time.time()-started,model_sha256=sha(run/'modelo.pt'),data_manifest_sha256=sha(DATA/'manifiesto.json'),coverage80=float(np.mean((v>=q[:,0])&(v<=q[:,2]))),width80_s=float(np.mean(q[:,2]-q[:,0])),conclusion=('Dos rutas proyectadas con tráfico medido, una hipótesis de tiempo libre y un componente OD residual. La mediana de la mezcla se contrasta con TabM directo con las mismas características. Los componentes son hipótesis de costo y no se interpretan todos como rutas distintas.' if a.feature_set=='projected' else 'Distribución condicionada por tres rutas calculadas y componente residual, con mediana de la mezcla. Contrastar con el control directo y la ablación de forma; sin trayectoria observada de consulta.'))
write_json(run/'resultado.json',rec);register(rec);print(json.dumps(rec),flush=True)

