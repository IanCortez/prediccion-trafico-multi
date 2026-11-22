from common import *
from features import *
from inferencia import feature_input
from modelo_tabm import CityTabM
import torch,argparse,random,rtdl_num_embeddings as rtdl
p=argparse.ArgumentParser();p.add_argument("--id",required=True);p.add_argument("--seed",type=int,default=0);p.add_argument("--k",type=int,default=16);p.add_argument("--width",type=int,default=512);p.add_argument("--batch",type=int,default=512);p.add_argument("--epochs",type=int,default=100);p.add_argument("--gps-routes",action="store_true")
p.add_argument("--map-geometry",action="store_true")
p.add_argument("--weekly-only",action="store_true")
p.add_argument("--dynamic-routes",action="store_true")
p.add_argument("--hourly-gps",action="store_true")
p.add_argument("--archive-maps",nargs="+",choices=["2014","2026"],default=[])
p.add_argument("--calendar",action="store_true")
p.add_argument("--projected-routes",action="store_true");p.add_argument("--mapmatched",action="store_true")
p.add_argument("--learned-route",action="store_true")
p.add_argument("--road-attributes",action="store_true")
a=p.parse_args();run=RUNS/a.id;run.mkdir(exist_ok=False)
cfg=vars(a).copy();cfg.update(routes=True,context=True,architecture="tabm");snapshot(run,cfg);started=time.time()
torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed);torch.set_float32_matmul_precision("high")
tr,X,names=feature_input("train",dict(config=cfg,architecture="tabm"));va,V,_=feature_input("validation",dict(config=cfg,architecture="tabm"))
y=np.load(DATA/"train_y.npy");v=np.load(DATA/"validation_y.npy")
keep=np.flatnonzero(np.ptp(X,axis=0)>1e-7)
mean=X[:,keep].mean(0,dtype=np.float64).astype(np.float32);std=np.maximum(X[:,keep].std(0,dtype=np.float64),1e-5).astype(np.float32)
normalized=(X[:,keep]-mean)/std
rng=np.random.default_rng(a.seed);ix=rng.choice(len(y),min(200000,len(y)),replace=False)
sample=np.vstack([normalized[ix],normalized.min(0),normalized.max(0)])
bins=rtdl.compute_bins(torch.as_tensor(sample),n_bins=48)
center=float(np.median(y));model=CityTabM(mean,std,keep,bins,center,a.k,a.width).cuda()
x=torch.tensor(X,device="cuda");vx=torch.tensor(V,device="cuda");target=torch.tensor(y,device="cuda")
opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0003)
sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=.5,patience=4,min_lr=1e-5)
@torch.no_grad()
def predict():
    model.eval();out=[]
    for z in vx.split(2048):
        with torch.autocast("cuda",dtype=torch.bfloat16):pred=model(z)
        out.append(pred.float().mean(1).cpu().numpy())
    return np.maximum(np.concatenate(out),1)
best=float("inf");bad=0;history=[]
for epoch in range(1,a.epochs+1):
    model.train();total=0
    for ix in torch.randperm(len(x),device="cuda").split(a.batch):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            pred=model(x[ix]);loss=(pred-target[ix,None]).abs().mean()/600
        # Independent member losses; no averaging of predictions in the training loss.
        if not torch.isfinite(loss):raise RuntimeError("Non-finite TabM loss")
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=loss.item()*len(ix)
    pred=predict();vm=metrics(v,pred);sched.step(vm["mae_s"])
    rec=dict(epoch=epoch,train_loss=total/len(x),validation=vm,seconds=time.time()-started);history.append(rec);print(json.dumps(rec),flush=True)
    if vm["mae_s"]<best-.02:
        best=vm["mae_s"];bad=0
        torch.save(dict(state=model.state_dict(),mean=mean,std=std,keep=keep,bins=[b.cpu() for b in bins],center=center,config=cfg,features=names),run/"modelo.pt")
        np.savez_compressed(run/"validacion.npz",row=va["row"],timestamp=va["timestamp"],pred=pred,true=v);write_json(run/"mejor.json",rec)
    else:bad+=1
    write_json(run/"historia.json",history)
    if bad>=15:break
pred=np.load(run/"validacion.npz")["pred"]
rec=dict(id=a.id,architecture="tabm",config=cfg,features=names,kept_numerical_features=len(keep),validation=metrics(v,pred),interval=block_interval(v,pred,va["timestamp"]),train_n=len(y),epochs=len(history),parameters=sum(w.numel() for w in model.parameters()),elapsed_seconds=time.time()-started,model_sha256=sha(run/"modelo.pt"),data_manifest_sha256=sha(DATA/"manifiesto.json"),reference="https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html",conclusion="Control neuronal moderno con ensamble eficiente y embeddings numéricos. Compara capacidad/optimización frente a atención y árboles con las mismas entradas; no es una aportación original de esta tesis.")
write_json(run/"resultado.json",rec);register(rec);print(json.dumps(rec),flush=True)








