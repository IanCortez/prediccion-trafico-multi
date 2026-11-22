from common import *
from features import *
from modelo_trayectorias import *
import argparse,random
p=argparse.ArgumentParser();p.add_argument("--id",required=True);p.add_argument("--pretrained");p.add_argument("--epochs",type=int,default=80);p.add_argument("--batch",type=int,default=2048);p.add_argument("--lr",type=float,default=.0005);p.add_argument("--seed",type=int,default=0);p.add_argument("--train-n",type=int,default=0)
p.add_argument("--context-grid",choices=["counts","semantic"],default="counts")
p.add_argument("--route-aux",type=float,default=0)
a=p.parse_args();run=RUNS/a.id;run.mkdir(exist_ok=False);cfg=vars(a).copy();cfg["architecture"]="trajectory_route_aux" if a.route_aux else "trajectory_transformer";snapshot(run,cfg);started=time.time()
torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed);torch.set_float32_matmul_precision("high")
tr,y,va,v=load_development()
if a.train_n:
    ix=np.sort(np.random.default_rng(a.seed).choice(len(y),min(a.train_n,len(y)),replace=False))
    tr={k:z[ix] for k,z in tr.items()};y=y[ix]
gridfile=DATA/("contexto_semantico_rejilla25m.npz" if a.context_grid=="semantic" else "contexto_rejilla25m.npz")
grid,info=load_grid(gridfile);coef=fit_power(tr,y)
model=(RouteAuxEncoder if a.route_aux else TrajectoryEncoder)(grid,info,coef).cuda()
if a.pretrained:
    ck=torch.load(a.pretrained,map_location="cpu",weights_only=False);warm_load(model,ck)
    model.coef.copy_(torch.tensor(coef,device="cuda"))
def od_tensors(d):
    f,_=features(d)
    xy=np.stack([f[:,:2],f[:,2:4]],axis=1)
    return torch.as_tensor(xy,device="cuda"),torch.as_tensor(f[:,18:24],device="cuda")
x,dep=od_tensors(tr);vx,vd=od_tensors(va);target=torch.tensor(y,device="cuda")
if a.route_aux:
    geo=np.load(DATA/"geometria_train.npz");pos=np.searchsorted(geo["row"],tr["row"]);safe=np.minimum(pos,len(geo["row"])-1)
    mask=(pos<len(geo["row"]))&(geo["row"][safe]==tr["row"])
    g=geo["X"][safe];mask&=np.isfinite(g).all(1)&(g[:,-1]<=500)
    aux=np.column_stack([g[:,1:15],np.log(np.maximum(g[:,0]/tr["od_m"],1)).clip(0,3)]).astype(np.float32)
    aux[~mask]=0;aux=torch.as_tensor(aux,device="cuda");auxmask=torch.as_tensor(mask,device="cuda")
@torch.no_grad()
def predict():
    model.eval();out=[]
    for xy,t in zip(vx.split(8192),vd.split(8192)):
        known=torch.zeros(xy.shape[:2],device="cuda");visible=torch.zeros_like(known,dtype=torch.bool);visible[:,0]=True
        with torch.autocast("cuda",dtype=torch.bfloat16):pred,_=model(xy,known,visible,t)
        out.append(pred[:,1].cpu().numpy())
    return np.concatenate(out)
opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.005)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=.5,patience=4,min_lr=1e-5)
best=float("inf");bad=0;history=[]
for epoch in range(1,a.epochs+1):
    model.train();order=torch.randperm(len(x),device="cuda");total=0
    for ix in order.split(a.batch):
        xy=x[ix];known=torch.zeros(xy.shape[:2],device="cuda");visible=torch.zeros_like(known,dtype=torch.bool);visible[:,0]=True
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            output=model(xy,known,visible,dep[ix],return_hidden=bool(a.route_aux))
            pred=output[0];loss=(pred[:,1]-target[ix]).abs().mean()/600
            if a.route_aux:
                ok=auxmask[ix]
                if ok.any():
                    route=model.route_head(output[2][ok,1]);truth=aux[ix][ok]
                    geometry_loss=torch.nn.functional.smooth_l1_loss(route[:,:14],truth[:,:14],beta=.25)
                    circuity_loss=torch.nn.functional.smooth_l1_loss(route[:,14],truth[:,14],beta=.25)
                    loss=loss+a.route_aux*(geometry_loss+.25*circuity_loss)
        if not torch.isfinite(loss):raise RuntimeError("Non-finite OD loss")
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=loss.item()*len(ix)
    pred=predict();vm=metrics(v,pred);scheduler.step(vm["mae_s"])
    rec=dict(epoch=epoch,train_loss=total/len(x),validation=vm,lr=opt.param_groups[0]["lr"],seconds=time.time()-started);history.append(rec);print(json.dumps(rec),flush=True)
    if vm["mae_s"]<best-.02:
        best=vm["mae_s"];bad=0
        torch.save(dict(state=model.state_dict(),coef=coef,grid_info=info,config=cfg,epoch=epoch),run/"modelo.pt")
        np.savez_compressed(run/"validacion.npz",row=va["row"],pred=pred,true=v,timestamp=va["timestamp"])
        write_json(run/"mejor.json",rec)
    else:bad+=1
    write_json(run/"historia.json",history)
    if bad>=15:break
pred=np.load(run/"validacion.npz")["pred"]
rec=dict(id=a.id,architecture=cfg["architecture"],config=cfg,validation=metrics(v,pred),interval=block_interval(v,pred,va["timestamp"]),diagnostic=diagnose(va,v,pred),train_n=len(y),epochs=len(history),best_epoch=json.loads((run/"mejor.json").read_text())["epoch"],parameters=sum(z.numel() for z in model.parameters()),elapsed_seconds=time.time()-started,model_sha256=sha(run/"modelo.pt"),pretrained_sha256=sha(a.pretrained) if a.pretrained else None,data_manifest_sha256=sha(DATA/"manifiesto.json"),context_grid_sha256=sha(gridfile),inference="Exactly two coordinates and departure cyclical features; no arrival, point count or observed path",conclusion="Compara reconstrucción enmascarada de trayectorias con la misma arquitectura entrenada directamente para OD. Separar ganancia de preentrenamiento de ganancia por capacidad.")
write_json(run/"resultado.json",rec);register(rec);print(json.dumps(rec),flush=True)



