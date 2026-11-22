from common import *
from features import *
from inferencia import feature_input
from modelo_mezcla_rutas import RouteMixtureTabM
import torch,argparse,random
p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('--source',required=True);p.add_argument('--pinball',type=float,required=True);p.add_argument('--epochs',type=int,default=30);p.add_argument('--seed',type=int,default=0);a=p.parse_args();assert a.pinball in (0.,3.)
source=RUNS/a.source;parent=json.loads((source/'resultado.json').read_text(encoding='utf-8-sig'));assert parent['architecture']=='tabm_route_mixture';assert parent['config']['aux']==0;assert sha(source/'modelo.pt')==parent['model_sha256'];assert json.loads((BASE/'instrumentos/cuantil_interpolado_contrato.json').read_text())['passed']
run=RUNS/a.id;run.mkdir(exist_ok=False);started=time.time();cfg=parent['config'].copy();cfg.update(id=a.id,seed=a.seed,epochs=a.epochs,pinball=a.pinball,initialization_source=a.source,median_mode='interpolated',inference_precision='float32',canonical_batch=2048,learning_rate=.0001,stage='Matched warm continuation with interpolated duration median');cfg.pop('trained_model_source',None);snapshot(run,cfg)
torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed);torch.set_float32_matmul_precision('highest');tr,X,names=feature_input('train',dict(config=cfg,architecture='tabm_route_mixture'));va,V,vnames=feature_input('validation',dict(config=cfg,architecture='tabm_route_mixture'));assert names==vnames==parent['features'];y=np.load(DATA/'train_y.npy');v=np.load(DATA/'validation_y.npy');ck=torch.load(source/'modelo.pt',map_location='cpu',weights_only=False);model=RouteMixtureTabM(ck['mean'],ck['std'],ck['keep'],ck['bins'],ck['anchors'],ck['anchor_indices'],cfg['k'],cfg['width']).cuda();model.load_state_dict(ck['state']);x=torch.tensor(X,device='cuda');vx=torch.tensor(V,device='cuda');target=torch.tensor(y,device='cuda');opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0003);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=3,min_lr=1e-5)
@torch.no_grad()
def predict():
    model.eval();out=[]
    for block in vx.split(2048):
        real_size=len(block)
        if real_size<2048:block=torch.cat([block,block[-1:].expand(2048-real_size,-1)],0)
        logp,_=model(block);out.append(model.quantiles(logp,mode='interpolated')[:real_size].cpu().numpy())
    return np.concatenate(out)
def save(epoch,q):
    saved={**ck,'state':model.state_dict(),'config':cfg,'features':names};torch.save(saved,run/'modelo.pt');np.savez_compressed(run/'validacion.npz',row=va['row'],timestamp=va['timestamp'],pred=q[:,1],quantiles=q,true=v);write_json(run/'mejor.json',dict(epoch=epoch,validation=metrics(v,q[:,1])))
q=predict();initial=metrics(v,q[:,1]);best=initial['mae_s'];save(0,q);history=[dict(epoch=0,validation=initial,seconds=time.time()-started)];bad=0;print(json.dumps(history[0]),flush=True)
for epoch in range(1,a.epochs+1):
    model.train();total=0.
    for ix in torch.randperm(len(x),device='cuda').split(512):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):logp,_=model(x[ix])
        labels=((target[ix]-60)/15).long();nll=-logp.gather(-1,labels[:,None,None].expand(-1,cfg['k'],1)).mean();desired=(model.duration_grid[None,None,:]>=target[ix,None,None]).float();rps=(logp.exp().cumsum(-1)-desired).square().mean();median=model.quantiles(logp,(.5,),mode='interpolated')[:,0];point=(median-target[ix]).abs().mean()/600;loss=nll+5*rps+a.pinball*point
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite warm continuation loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=float(loss.detach())*len(ix)
    q=predict();vm=metrics(v,q[:,1]);scheduler.step(vm['mae_s']);r=dict(epoch=epoch,train_loss=total/len(x),validation=vm,coverage80=float(np.mean((v>=q[:,0])&(v<=q[:,2]))),width80_s=float(np.mean(q[:,2]-q[:,0])),seconds=time.time()-started);history.append(r);print(json.dumps(r),flush=True)
    if vm['mae_s']<best-.02:best=vm['mae_s'];bad=0;save(epoch,q)
    else:bad+=1
    write_json(run/'historia.json',history)
    if bad>=8:break
z=np.load(run/'validacion.npz');q=z['quantiles'];rec=dict(id=a.id,architecture='tabm_route_mixture',config=cfg,features=names,anchor_features=parent['anchor_features'],validation=metrics(v,z['pred']),interval=block_interval(v,z['pred'],va['timestamp']),train_n=len(y),epochs=len(history)-1,parameters=sum(p.numel() for p in model.parameters()),elapsed_seconds=time.time()-started,initialization_source=a.source,source_model_sha256=parent['model_sha256'],initial_validation=initial,model_sha256=sha(run/'modelo.pt'),data_manifest_sha256=sha(DATA/'manifiesto.json'),coverage80=float(np.mean((v>=q[:,0])&(v<=q[:,2]))),width80_s=float(np.mean(q[:,2]-q[:,0])),conclusion='Continuación pareada desde un checkpoint común, con mediana interpolada y evaluación float32. Comparar pinball3 contra pinball0 para separar objetivo puntual y entrenamiento adicional. La interpolación supone masa uniforme dentro de cada bin, no la mediana continua exacta de componentes lognormales.');write_json(run/'resultado.json',rec);register(rec);print(json.dumps(dict(id=a.id,validation=rec['validation'],initial=initial)),flush=True)
