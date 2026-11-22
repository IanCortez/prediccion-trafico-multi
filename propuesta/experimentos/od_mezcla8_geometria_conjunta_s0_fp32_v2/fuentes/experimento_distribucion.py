from common import *
from features import *
from modelo_trayectorias import *
import argparse,random
p=argparse.ArgumentParser();p.add_argument("--id",required=True);p.add_argument("--pretrained");p.add_argument("--seed",type=int,default=0);p.add_argument("--epochs",type=int,default=90);p.add_argument("--rps",type=float,default=5);p.add_argument("--context-grid",choices=["counts","semantic"],default="semantic")
a=p.parse_args();run=RUNS/a.id;run.mkdir(exist_ok=False);cfg=vars(a).copy();cfg["architecture"]="duration_distribution";snapshot(run,cfg);started=time.time()
torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed);torch.set_float32_matmul_precision("high")
tr,y,va,v=load_development();coef=fit_power(tr,y)
gridfile=DATA/("contexto_semantico_rejilla25m.npz" if a.context_grid=="semantic" else "contexto_rejilla25m.npz")
grid,info=load_grid(gridfile);model=DurationClassifier(grid,info,coef).cuda()
if a.pretrained:
    ck=torch.load(a.pretrained,map_location="cpu",weights_only=False);warm_load(model,ck);model.coef.copy_(torch.tensor(coef,device="cuda"))
def tensors(d):
    f,_=features(d);return torch.tensor(np.stack([f[:,:2],f[:,2:4]],1),device="cuda"),torch.tensor(f[:,18:24],device="cuda")
x,t=tensors(tr);vx,vt=tensors(va);target=torch.as_tensor(((y-60)/15).astype(np.int64),device="cuda")
opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.005);sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=.5,patience=4,min_lr=1e-5)
@torch.no_grad()
def predict():
    model.eval();out=[]
    for xx,tt in zip(vx.split(8192),vt.split(8192)):
        with torch.autocast("cuda",dtype=torch.bfloat16):logits=model.posterior(xx,tt)
        out.append(model.quantiles(logits).cpu().numpy())
    return np.concatenate(out)
best=float("inf");bad=0;history=[]
for epoch in range(1,a.epochs+1):
    model.train();total=0
    for ix in torch.randperm(len(x),device="cuda").split(2048):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            logits=model.posterior(x[ix],t[ix]);ce=torch.nn.functional.cross_entropy(logits,target[ix])
            cdf=logits.softmax(-1).cumsum(-1);truth=torch.arange(116,device="cuda")[None,:]>=target[ix,None]
            rps=(cdf-truth.float()).square().mean();loss=ce+a.rps*rps
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=loss.item()*len(ix)
    q=predict();pred=q[:,1];vm=metrics(v,pred);sched.step(vm["mae_s"])
    rec=dict(epoch=epoch,train_loss=total/len(x),validation=vm,coverage80=float(((v>=q[:,0])&(v<=q[:,2])).mean()),width80_s=float((q[:,2]-q[:,0]).mean()),seconds=time.time()-started)
    history.append(rec);print(json.dumps(rec),flush=True)
    if vm["mae_s"]<best-.02:
        best=vm["mae_s"];bad=0;torch.save(dict(state=model.state_dict(),config=cfg,grid_info=info,coef=coef),run/"modelo.pt")
        np.savez_compressed(run/"validacion.npz",row=va["row"],timestamp=va["timestamp"],pred=pred,true=v,q10=q[:,0],q90=q[:,2]);write_json(run/"mejor.json",rec)
    else:bad+=1
    write_json(run/"historia.json",history)
    if bad>=15:break
z=np.load(run/"validacion.npz");pred=z["pred"]
rec=dict(id=a.id,architecture="duration_distribution",config=cfg,validation=metrics(v,pred),interval=block_interval(v,pred,va["timestamp"]),train_n=len(y),epochs=len(history),parameters=sum(w.numel() for w in model.parameters()),elapsed_seconds=time.time()-started,model_sha256=sha(run/"modelo.pt"),context_grid_sha256=sha(gridfile),coverage80=float(((v>=z["q10"])&(v<=z["q90"])).mean()),width80_s=float((z["q90"]-z["q10"]).mean()),data_manifest_sha256=sha(DATA/"manifiesto.json"),conclusion="Evalúa si representar la distribución multimodal y ordenada mejora su mediana frente a regresión escalar. Intervalos descriptivos sin garantía de cobertura tras cambio temporal.")
write_json(run/"resultado.json",rec);register(rec);print(json.dumps(rec),flush=True)

