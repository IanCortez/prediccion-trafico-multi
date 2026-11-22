from common import *
from features import *
from modelo_trayectorias import *
import argparse,random
p=argparse.ArgumentParser();p.add_argument("--id",required=True);p.add_argument("--epochs",type=int,default=5);p.add_argument("--batch",type=int,default=256);p.add_argument("--seed",type=int,default=0);p.add_argument("--max-trips",type=int,default=0);p.add_argument("--resume")
a=p.parse_args();run=RUNS/a.id
if a.resume:assert run.exists()
else:run.mkdir(exist_ok=False)
snapshot(run,vars(a));started=time.time()
torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
torch.set_float32_matmul_precision("high")
tr,y,va,v=load_development();coef=fit_power(tr,y)
grid,info=load_grid(DATA/"contexto_rejilla25m.npz")
model=TrajectoryEncoder(grid,info,coef).cuda()
opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
z=np.load(DATA/"trayectorias_train.npz");xy=torch.tensor(z["xy"],device="cuda");offsets=torch.tensor(z["offsets"],device="cuda")
length=offsets[1:]-offsets[:-1]
fake={k:np.zeros(len(z["timestamp"])) for k in ("o_lon","o_lat","d_lon","d_lat","od_m")}
fake["od_m"]+=200;fake["timestamp"]=z["timestamp"]
dep=torch.tensor(features(fake)[0][:,18:24],device="cuda")
n=len(length);n=min(n,a.max_trips) if a.max_trips else n
initial=0
if a.resume:
    ck=torch.load(a.resume,map_location="cpu",weights_only=False);model.load_state_dict(ck["state"]);opt.load_state_dict(ck["optimizer"]);initial=ck["epoch"]
    torch.set_rng_state(ck["rng_cpu"]);torch.cuda.set_rng_state(ck["rng_cuda"])
history=[]
for epoch in range(initial+1,a.epochs+1):
    model.train();order=torch.randperm(n,device="cuda")
    # Random groups sorted by length reduce padding without ordering labels.
    order=torch.cat([ix[length[ix].argsort()] for ix in order.split(4096)])
    totals=np.zeros(3);seen=0
    for step,ix in enumerate(order.split(a.batch),1):
        B=len(ix);L=int(length[ix].max());j=torch.arange(L,device="cuda")[None,:].expand(B,-1)
        valid=j<length[ix,None]
        keep=(torch.rand(B,L,device="cuda")<(.3+.7*torch.rand(B,1,device="cuda")))&valid
        keep[:,0]=True;keep.scatter_(1,(length[ix]-1)[:,None],True)
        kept=keep.sum(1);newL=int(kept.max())
        selected=torch.where(keep,j,j+L).argsort(1)[:,:newL]
        pad=torch.arange(newL,device="cuda")[None,:]>=kept[:,None]
        positions=(offsets[ix,None]+selected).clamp_max(len(xy)-1)
        points=xy[positions].masked_fill(pad[...,None],0)
        elapsed=selected.float()*15
        visible=(torch.rand(B,newL,device="cuda")>.5)&~pad;visible[:,0]=True
        spatial_visible=(torch.rand(B,newL,device="cuda")>.15)&~pad;spatial_visible[:,0]=True
        target_mask=~visible&~pad&spatial_visible
        space_mask=~spatial_visible&~pad
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            pred,sp=model(points,elapsed,visible,dep[ix],pad,spatial_visible)
            lt=((pred-elapsed).abs()*target_mask).sum()/target_mask.sum().clamp_min(1)/600
            rel=(points-points[:,:1])/5
            ls=((sp-rel).square().sum(-1)*space_mask).sum()/space_mask.sum().clamp_min(1)
            loss=lt+.1*ls
        if not torch.isfinite(loss):raise RuntimeError("Non-finite pretraining loss")
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        totals+=np.array([loss.item(),lt.item(),ls.item()])*B;seen+=B
        if step%250==0:
            out=dict(epoch=epoch,step=step,seen=seen,losses=(totals/seen).tolist(),seconds=time.time()-started)
            print(json.dumps(out),flush=True);write_json(run/"progreso.json",out)
    record=dict(epoch=epoch,losses=(totals/seen).tolist(),n=seen,seconds=time.time()-started);history.append(record)
    ck=dict(state=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,coef=coef,grid_info=info,config=vars(a),rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(),architecture="trajectory_transformer")
    torch.save(ck,run/"ultimo.pt")
    if epoch in (1,5,10,20,30):torch.save(ck,run/f"epoca_{epoch}.pt")
    write_json(run/"historia_preentrenamiento.json",history);print(json.dumps(record),flush=True)
write_json(run/"resultado_preentrenamiento.json",dict(id=a.id,n_trips=n,points=int(z["offsets"][n]),epochs=a.epochs,parameters=sum(q.numel() for q in model.parameters()),elapsed_seconds=time.time()-started,checkpoint_sha256=sha(run/"ultimo.pt"),only_train=True,thinning="random 30%-100%, retain first/last; original indices never enter encoder",test_labels_read=False))

