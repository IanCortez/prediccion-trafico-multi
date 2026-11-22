from common import *
from features import *
from modelo_residual import build_model
import argparse,torch,random
p=argparse.ArgumentParser();p.add_argument("--id",required=True);p.add_argument("--seed",type=int,default=0);p.add_argument("--width",type=int,default=256);p.add_argument("--epochs",type=int,default=140);p.add_argument("--batch",type=int,default=4096);p.add_argument("--lr",type=float,default=.001);p.add_argument("--dropout",type=float,default=.05);p.add_argument("--no-grids",action="store_true");p.add_argument("--no-fourier",action="store_true");p.add_argument("--enriched",action="store_true");p.add_argument("--loss",choices=["mae","loghuber"],default="mae");p.add_argument("--train-n",type=int,default=0)
p.add_argument("--routes",action="store_true");p.add_argument("--pretrain-epochs",type=int,default=0)
p.add_argument("--context",action="store_true");p.add_argument("--architecture",choices=["residual","film"],default="residual");p.add_argument("--network-anchor",action="store_true")
a=p.parse_args();run=RUNS/a.id;run.mkdir(exist_ok=False);started=time.time();snapshot(run,vars(a))
torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
torch.backends.cudnn.benchmark=False
tr,y,va,v=load_development();X,names=features(tr,a.enriched);V,_=features(va,a.enriched)
if a.routes:
    rr=np.load(DATA/"train_rutas.npz");vv=np.load(DATA/"validation_rutas.npz")
    assert np.array_equal(rr["row"],tr["row"]) and np.array_equal(vv["row"],va["row"])
    X=np.column_stack([X,rr["X"]]);V=np.column_stack([V,vv["X"]]);names+=rr["names"].tolist()
if a.context:
    rr=np.load(DATA/"train_contexto.npz");vv=np.load(DATA/"validation_contexto.npz")
    assert np.array_equal(rr["row"],tr["row"]) and np.array_equal(vv["row"],va["row"])
    X=np.column_stack([X,rr["X"]]);V=np.column_stack([V,vv["X"]]);names+=rr["names"].tolist()
if a.train_n:
    ix=np.sort(np.random.default_rng(a.seed).choice(len(y),min(a.train_n,len(y)),replace=False))
    tr={k:z[ix] for k,z in tr.items()};X=X[ix];y=y[ix]
mean=X.mean(0);std=np.maximum(X.std(0),1e-3);coef=fit_power(tr,y)
if a.network_anchor:
    assert a.routes and a.architecture=="film"
    Z=np.column_stack([np.ones(len(y)),np.log(tr["od_m"].astype(np.float64)),np.log(np.maximum(X[:,33].astype(np.float64)*600,1))])
    coef=np.linalg.lstsq(Z,np.log(y.astype(np.float64)),rcond=None)[0]
    coef[0]+=np.log(np.median(y/np.exp(Z@coef)))
    coef=coef.tolist()
device="cuda";model=build_model(mean,std,coef,vars(a)).to(device)
x=torch.tensor(X,device=device);target=torch.tensor(y,device=device);vx=torch.tensor(V,device=device)
if a.pretrain_epochs:
    assert not a.routes and not a.context and not a.enriched and not a.train_n,"Separate augmentation comparison needs same inputs and complete training cohort"
    au=dict(np.load(DATA/"subtrayectos.npz"));AX,_=features(au)
    ax=torch.tensor(AX,device=device);ay=torch.tensor(au["duration"],device=device)
opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=.5,patience=4,min_lr=1e-5)
best=float("inf");bad=0;history=[]
@torch.no_grad()
def predict():
    model.eval()
    return torch.cat([model(z) for z in vx.split(8192)]).cpu().numpy()
for epoch in range(1,a.epochs+1):
    model.train();epoch_x,epoch_y=(ax,ay) if epoch<=a.pretrain_epochs else (x,target)
    order=torch.randperm(len(epoch_x),device=device);total=0
    for ix in order.split(a.batch):
        opt.zero_grad(set_to_none=True);pred=model(epoch_x[ix])
        if a.loss=="mae":loss=(pred-epoch_y[ix]).abs().mean()/600
        else:loss=torch.nn.functional.smooth_l1_loss(pred.log(),epoch_y[ix].log(),beta=.25)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=loss.item()*len(ix)
    pred=predict();vm=metrics(v,pred);sched.step(vm["mae_s"])
    rec=dict(epoch=epoch,phase="subtrajectory" if epoch<=a.pretrain_epochs else "journey",train_loss=total/len(epoch_x),validation=vm,lr=opt.param_groups[0]["lr"],seconds=time.time()-started);history.append(rec)
    print(json.dumps(rec),flush=True)
    if vm["mae_s"]<best-.02:
        best=vm["mae_s"];bad=0
        torch.save(dict(state=model.state_dict(),mean=mean,std=std,coef=coef,config=vars(a),features=names),run/"modelo.pt")
        np.savez_compressed(run/"validacion.npz",row=va["row"],pred=pred,true=v,timestamp=va["timestamp"])
        write_json(run/"mejor.json",rec)
    else:bad+=1
    write_json(run/"historia.json",history)
    if epoch==a.pretrain_epochs:
        for group in opt.param_groups:group["lr"]=a.lr/2
        sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=.5,patience=4,min_lr=1e-5)
        bad=0
    if bad>=18 and epoch>a.pretrain_epochs:break
saved=np.load(run/"validacion.npz");pred=saved["pred"]
rec=dict(id=a.id,architecture="OD physical residual multiscale",config=vars(a),power_coefficients=coef,validation=metrics(v,pred),interval=block_interval(v,pred,va["timestamp"]),diagnostic=diagnose(va,v,pred),train_n=len(y),validation_n=len(v),features=names,parameters=sum(z.numel() for z in model.parameters()),epochs=len(history),best_epoch=json.loads((run/"mejor.json").read_text())["epoch"],elapsed_seconds=time.time()-started,model_sha256=sha(run/"modelo.pt"),data_manifest_sha256=sha(DATA/"manifiesto.json"),code_sha256=sha(__file__))
rec["conclusion"]="Residuo físico con MAE en segundos y resolución espacial fina; comparar ganancia y complejidad con árbol y ablaciones antes de aceptar."
write_json(run/"resultado.json",rec);register(rec);print(json.dumps(rec,ensure_ascii=False),flush=True)




