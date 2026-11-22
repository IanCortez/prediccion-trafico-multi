from common import *
from features import *
from inferencia import feature_input
from modelo_mezcla_conjunta import JointRouteTimeMixture
import torch,argparse,random,rtdl_num_embeddings as rtdl
p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('--geometry-weight',type=float,required=True,choices=[0.,.2]);p.add_argument('--pause-weight',type=float,default=0.,choices=[0.,.2]);p.add_argument('--seed',type=int,default=0);p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch',type=int,default=512);p.add_argument('--k',type=int,default=16);p.add_argument('--width',type=int,default=512);p.add_argument('--verify-only',action='store_true');a=p.parse_args();assert not (a.pause_weight and a.geometry_weight),'Separate auxiliary targets in the matched ablations';cfg=vars(a).copy();cfg.update(architecture='tabm_joint_route_time',routes=True,context=True,projected_routes=True,mapmatched=True,calendar=True,components=8,median_mode='discrete',auxiliary_kind='low_displacement_counts' if a.pause_weight else 'route_geometry' if a.geometry_weight else 'none');started=time.time()
run=RUNS/a.id
if a.pause_weight:cfg['pause_target_sha256']=sha(DATA/'pausas_gps_train.npz')
if not a.verify_only:run.mkdir(exist_ok=False);snapshot(run,cfg)
torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed);torch.set_float32_matmul_precision('high');tr,X,names=feature_input('train',dict(config=cfg,architecture=cfg['architecture']));va,V,_=feature_input('validation',dict(config=cfg,architecture=cfg['architecture']));y=np.load(DATA/'train_y.npy');v=np.load(DATA/'validation_y.npy');keep=np.flatnonzero(np.ptp(X,axis=0)>1e-7);mean=X[:,keep].mean(0,dtype=np.float64).astype(np.float32);std=np.maximum(X[:,keep].std(0,dtype=np.float64),1e-5).astype(np.float32);normalized=(X[:,keep]-mean)/std;ix=np.random.default_rng(a.seed).choice(len(y),min(200000,len(y)),replace=False);sample=np.vstack([normalized[ix],normalized.min(0),normalized.max(0)]);bins=rtdl.compute_bins(torch.as_tensor(sample),n_bins=48)
anchor_names=('mm_length_gps_time_600s','mm_free_gps_time_600s','projected_free_freeflow_600s','od_km');anchor_indices=[names.index(s) for s in anchor_names];anchors=[]
for col,scale in zip(anchor_indices,(600,600,600,1000)):
    logx=np.log(np.maximum(X[:,col].astype(float)*scale,.1));beta,intercept=np.polyfit(logx,np.log(y),1);intercept+=np.log(np.median(y/np.exp(intercept+beta*logx)));anchors.append([intercept,beta])
short=[names.index('projected_length_shape_'+str(i)) for i in range(14)];fast=[names.index('projected_free_shape_'+str(i)) for i in range(14)];geometry_indices=np.array([short,fast,fast,[-1]*14]*2);model=JointRouteTimeMixture(mean,std,keep,bins,anchors,anchor_indices,geometry_indices,a.k,a.width).cuda();gt=dict(np.load(DATA/'geometria_train.npz'));pos=np.searchsorted(gt['row'],tr['row']);safe=np.minimum(pos,len(gt['row'])-1);valid=(pos<len(gt['row']))&(gt['row'][safe]==tr['row'])&(gt['X'][safe,-1]<=500);geometry=gt['X'][safe,1:15].copy();valid&=np.isfinite(geometry).all(1);geometry[~valid]=0

pause_counts=np.zeros(len(y),np.float32);pause_valid=np.zeros(len(y),bool)
if a.pause_weight:
    pauses=dict(np.load(DATA/'pausas_gps_train.npz'));pp=np.searchsorted(pauses['row'],tr['row']);safe_p=np.minimum(pp,len(pauses['row'])-1);pause_valid=(pp<len(pauses['row']))&(pauses['row'][safe_p]==tr['row'])&pauses['reliable'][safe_p];pause_counts=pauses['low_displacement_seconds'][safe_p,1]/15;pause_counts[~pause_valid]=0;assert np.all((pause_counts>=0)&(pause_counts<=y/15));cfg['pause_target_sha256']=sha(DATA/'pausas_gps_train.npz')
def loss_fn(output,target,g,mask,pause_target=None,pause_mask=None):
    logp=output[0];label=((target-60)/15).round().long();nll=-logp.gather(-1,label[:,None,None].expand(-1,a.k,1)).mean();desired=(model.duration_grid[None,None,:]>=target[:,None,None]).float();rps=(logp.exp().cumsum(-1)-desired).square().mean();shape=model.geometry_score(output,target,g,mask);pause=model.pause_score(output,target,pause_target,pause_mask) if a.pause_weight else 0.;return nll+5*rps+a.geometry_weight*shape+a.pause_weight*pause
if a.verify_only:
    model.eval();torch.set_float32_matmul_precision('highest');inputs=torch.tensor(X[:32],device='cuda');target=torch.tensor(y[:32],device='cuda');g=torch.tensor(geometry[:32],device='cuda');mask=torch.tensor(valid[:32],device='cuda',dtype=torch.float32);out=model(inputs);loss=loss_fn(out,target,g,mask,torch.tensor(pause_counts[:32],device='cuda'),torch.tensor(pause_valid[:32],device='cuda',dtype=torch.float32));assert torch.isfinite(loss);assert (out[0].exp().sum(-1)-1).abs().max()<1e-5;loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    with torch.no_grad():rev=model(inputs.flip(0))[0].flip(0)
    error=float((out[0].detach().exp()-rev.exp()).abs().max());assert error<1e-5;write_json(BASE/('instrumentos/mezcla_conjunta_forward_pausas.json' if a.pause_weight else 'instrumentos/mezcla_conjunta_forward.json'),dict(passed=True,geometry_weight=a.geometry_weight,pause_weight=a.pause_weight,model_code_sha256=sha(BASE/'modelo_mezcla_conjunta.py'),training_code_sha256=sha(BASE/'experimento_mezcla_conjunta.py'),features=len(names),components=8,finite_gradients=True,max_permutation_probability_error=error,geometry_valid_training_n=int(valid.sum()),query_geometry_not_input=True,test_labels_read=False));print('Joint route-time forward contracts passed',flush=True);sys.exit(0)
x=torch.tensor(X,device='cuda');vx=torch.tensor(V,device='cuda');target=torch.tensor(y,device='cuda');g=torch.tensor(geometry,device='cuda');mask=torch.tensor(valid,device='cuda',dtype=torch.float32);pause_target=torch.tensor(pause_counts,device='cuda');pause_mask=torch.tensor(pause_valid,device='cuda',dtype=torch.float32);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0003);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=5,min_lr=1e-5)
@torch.no_grad()
def predict():
    model.eval();out=[]
    for block in vx.split(2048):
        with torch.autocast('cuda',dtype=torch.bfloat16):value=model(block)
        out.append(model.quantiles(value[0]).cpu().numpy())
    return np.concatenate(out)
best=float('inf');bad=0;history=[]
for epoch in range(1,a.epochs+1):
    model.train();total=0.
    for ix in torch.randperm(len(x),device='cuda').split(a.batch):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):output=model(x[ix])
        loss=loss_fn(output,target[ix],g[ix],mask[ix],pause_target[ix],pause_mask[ix]);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=float(loss.detach())*len(ix)
    q=predict();vm=metrics(v,q[:,1]);scheduler.step(vm['mae_s']);r=dict(epoch=epoch,train_loss=total/len(x),validation=vm,coverage80=float(np.mean((v>=q[:,0])&(v<=q[:,2]))),width80_s=float(np.mean(q[:,2]-q[:,0])),seconds=time.time()-started);history.append(r);print(json.dumps(r),flush=True)
    if vm['mae_s']<best-.02:
        best=vm['mae_s'];bad=0;torch.save(dict(state=model.state_dict(),mean=mean,std=std,keep=keep,bins=[b.cpu() for b in bins],anchors=anchors,anchor_indices=anchor_indices,geometry_indices=geometry_indices,config=cfg,features=names),run/'modelo.pt');np.savez_compressed(run/'validacion.npz',row=va['row'],timestamp=va['timestamp'],pred=q[:,1],quantiles=q,true=v);write_json(run/'mejor.json',r)
    else:bad+=1
    write_json(run/'historia.json',history)
    if bad>=20:break
z=np.load(run/'validacion.npz');q=z['quantiles'];rec=dict(id=a.id,architecture=cfg['architecture'],config=cfg,features=names,anchor_features=anchor_names,validation=metrics(v,z['pred']),interval=block_interval(v,z['pred'],va['timestamp']),train_n=len(y),geometry_valid_training_n=int(valid.sum()),pause_valid_training_n=int(pause_valid.sum()),epochs=len(history),parameters=sum(p.numel() for p in model.parameters()),elapsed_seconds=time.time()-started,model_sha256=sha(run/'modelo.pt'),data_manifest_sha256=sha(DATA/'manifiesto.json'),coverage80=float(np.mean((v>=q[:,0])&(v<=q[:,2]))),width80_s=float(np.mean(q[:,2]-q[:,0])),conclusion=('Ocho componentes de duración con NLL beta-binomial condicional de intervalos GPS de bajo desplazamiento. El número de intervalos se deriva de la duración etiquetada solo en la pérdida; no es entrada de consulta. Misma arquitectura e inicialización que el control de peso auxiliar cero.' if a.pause_weight else 'Ocho componentes latentes de tiempo y geometría. La forma observada se usa solo como objetivo auxiliar condicional durante entrenamiento; inferencia marginaliza componentes con entradas OD. Comparar peso0.2 con control0 de arquitectura, datos y presupuesto iguales.'));write_json(run/'resultado.json',rec);register(rec);print(json.dumps(dict(id=a.id,validation=rec['validation'])),flush=True)
