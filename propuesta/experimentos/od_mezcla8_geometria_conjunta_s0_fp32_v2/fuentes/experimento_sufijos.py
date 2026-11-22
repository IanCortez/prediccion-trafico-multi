from common import *
from features import *
from modelo_trayectorias import *
import argparse,pandas as pd,random
p=argparse.ArgumentParser();p.add_argument("--id",required=True);p.add_argument("--seed",type=int,default=0);p.add_argument("--pretrained");p.add_argument("--pre-epochs",type=int,default=10);p.add_argument("--fine-epochs",type=int,default=60);p.add_argument("--no-adapter",action="store_true")
a=p.parse_args();run=RUNS/a.id;run.mkdir(exist_ok=False);cfg=vars(a).copy();cfg.update(architecture="dense_suffix",context_grid="semantic",task_adapter=not a.no_adapter);snapshot(run,cfg);started=time.time()
torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed);torch.set_float32_matmul_precision("high")
tr,y,va,v=load_development();coef=fit_power(tr,y);grid,info=load_grid(DATA/"contexto_semantico_rejilla25m.npz")
cls=TrajectoryEncoder if a.no_adapter else TaskAwareEncoder;model=cls(grid,info,coef).cuda()
if a.pretrained:
    warm_load(model,torch.load(a.pretrained,map_location="cpu",weights_only=False));model.coef.copy_(torch.tensor(coef,device="cuda"))
def tensors(d):
    f,_=features(d);return torch.tensor(np.stack([f[:,:2],f[:,2:4]],1),device="cuda"),torch.tensor(f[:,18:24],device="cuda")
x,t=tensors(tr);vx,vt=tensors(va);target=torch.tensor(y,device="cuda")
pack=np.load(DATA/"trayectorias_train.npz");xyall=torch.tensor(pack["xy"],device="cuda");offsets=torch.tensor(pack["offsets"],device="cuda")
length=offsets[1:]-offsets[:-1];ts=torch.tensor(pack["timestamp"],device="cuda");npk=len(length);n=len(y)
lo=pd.Timestamp(int(pack["timestamp"].min()),unit="s",tz="UTC").floor("D");hi=pd.Timestamp(int(pack["timestamp"].max())+1800,unit="s",tz="UTC").ceil("D")
hours=pd.date_range(lo,hi,freq="h");utc=hours.as_unit("s").asi8;wall=hours.tz_convert("Europe/Lisbon").tz_localize(None).as_unit("s").asi8;off=wall-utc
starts=np.r_[0,np.flatnonzero(np.diff(off)!=0)+1];cuts=torch.tensor(utc[starts],device="cuda");tzoff=torch.tensor(off[starts],device="cuda")
day0=int(wall.min()//86400)-1;days=np.arange(day0,int(wall.max()//86400)+3);month=np.asarray(pd.to_datetime(days*86400,unit="s").month)-1
season=torch.tensor(np.column_stack([np.sin(2*np.pi*month/12),np.cos(2*np.pi*month/12)]).astype(np.float32),device="cuda")
def departure(instant):
    local=instant+tzoff[(torch.bucketize(instant,cuts,right=True)-1).clamp_min(0)]
    day=local//86400;sec=local%86400;hourangle=sec.float()*(2*np.pi/86400)
    weekangle=(((day+3)%7)*86400+sec).float()*(2*np.pi/604800)
    return torch.cat([torch.stack([hourangle.sin(),hourangle.cos(),weekangle.sin(),weekangle.cos()],1),season[day-day0]],1)
def suffix(src):
    L=length[src];start=(torch.rand(len(src),device="cuda")*(L-4)).long();dest=xyall[offsets[src+1]-1]
    for attempt in range(5):
        current=xyall[offsets[src]+start];remaining=(L-1-start).float()*15
        distance=(dest-current).square().sum(1).sqrt()*1000
        nxt=xyall[offsets[src]+torch.minimum(start+1,L-1)]
        prev=xyall[offsets[src]+torch.maximum(start-1,torch.zeros_like(start))]
        jump=torch.maximum((nxt-current).square().sum(1).sqrt(),(prev-current).square().sum(1).sqrt())*1000
        good=(distance>=200)&(distance/remaining<=120/3.6)&(jump<=500)
        if good.all():break
        start[~good]=(torch.rand(int((~good).sum()),device="cuda")*(L[~good]-4)).long()
    start[~good]=0
    current=xyall[offsets[src]+start];remaining=(L-1-start).float()*15
    return torch.stack([current,dest],1),departure(ts[src]+start*15),remaining,(start>0).long()
@torch.no_grad()
def predict():
    model.eval();out=[]
    for xx,tt in zip(vx.split(8192),vt.split(8192)):
        known=torch.zeros(xx.shape[:2],device="cuda");vis=torch.zeros_like(known,dtype=torch.bool);vis[:,0]=True
        with torch.autocast("cuda",dtype=torch.bfloat16):pred,_=model(xx,known,vis,tt)
        out.append(pred[:,1].cpu().numpy())
    return np.concatenate(out)
def optimizer(lr):
    op=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.005)
    return op,torch.optim.lr_scheduler.ReduceLROnPlateau(op,mode="min",factor=.5,patience=4,min_lr=1e-5)
opt,sched=optimizer(.0005);best=float("inf");bad=0;history=[];presentations=0
for epoch in range(1,a.pre_epochs+a.fine_epochs+1):
    pre=epoch<=a.pre_epochs
    if epoch==a.pre_epochs+1:
        if (run/"modelo.pt").exists():model.load_state_dict(torch.load(run/"modelo.pt",map_location="cpu",weights_only=False)["state"])
        opt,sched=optimizer(.00025);bad=0
    model.train();total=0;seen=0
    size=n+2*npk if pre else n
    for ix in torch.randperm(size,device="cuda").split(2048):
        if pre:
            B=len(ix);xx=torch.empty((B,2,2),device="cuda");tt=torch.empty((B,6),device="cuda");yy=torch.empty(B,device="cuda");task=torch.zeros(B,dtype=torch.long,device="cuda");w=torch.ones(B,device="cuda")
            original=ix<n;xx[original]=x[ix[original]];tt[original]=t[ix[original]];yy[original]=target[ix[original]]
            source=(ix[~original]-n)%npk
            if len(source):
                sx,st,sy,phase=suffix(source);xx[~original]=sx;tt[~original]=st;yy[~original]=sy;task[~original]=phase;w[~original]=.5
        else:xx=x[ix];tt=t[ix];yy=target[ix];task=torch.zeros(len(ix),dtype=torch.long,device="cuda");w=torch.ones(len(ix),device="cuda")
        known=torch.zeros(xx.shape[:2],device="cuda");vis=torch.zeros_like(known,dtype=torch.bool);vis[:,0]=True
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            pred,_=model(xx,known,vis,tt,task_id=task);loss=((pred[:,1]-yy).abs()*w).sum()/w.sum()/600
        if not torch.isfinite(loss):raise RuntimeError("Non-finite suffix loss")
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=loss.item()*len(ix);seen+=len(ix)
    presentations+=seen;pred=predict();vm=metrics(v,pred);sched.step(vm["mae_s"])
    rec=dict(epoch=epoch,phase="mixed_whole_suffix" if pre else "whole_only",train_loss=total/seen,validation=vm,seconds=time.time()-started);history.append(rec);print(json.dumps(rec),flush=True)
    if vm["mae_s"]<best-.02:
        best=vm["mae_s"];bad=0;torch.save(dict(state=model.state_dict(),config=cfg,coef=coef,grid_info=info),run/"modelo.pt")
        np.savez_compressed(run/"validacion.npz",row=va["row"],timestamp=va["timestamp"],pred=pred,true=v);write_json(run/"mejor.json",rec)
    else:bad+=1
    write_json(run/"historia.json",history)
    if not pre and bad>=15:break
pred=np.load(run/"validacion.npz")["pred"]
rec=dict(id=a.id,architecture="dense_suffix",config=cfg,validation=metrics(v,pred),interval=block_interval(v,pred,va["timestamp"]),train_n=len(y),training_trajectories_with_internal_gps=npk,training_presentations=presentations,epochs=len(history),parameters=sum(z.numel() for z in model.parameters()),elapsed_seconds=time.time()-started,model_sha256=sha(run/"modelo.pt"),data_manifest_sha256=sha(DATA/"manifiesto.json"),inference="Two endpoints and departure; task prompt fixed to whole trip. No observed query path/count.",conclusion="Transferencia entre etiquetas de tiempo restante y viaje completo, con adaptador de tarea y ponderación de ejemplos; comparar contra mezcla sin adaptador y contra preentrenamiento aleatorio.")
write_json(run/"resultado.json",rec);register(rec);print(json.dumps(rec),flush=True)

