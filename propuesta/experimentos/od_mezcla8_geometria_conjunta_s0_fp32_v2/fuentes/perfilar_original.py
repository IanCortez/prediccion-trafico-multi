from common import *
import torch,time,sys,types
sys.path.insert(0,str(ROOT/"codigo-ian/src"))
import tte_model as M
torch.set_num_threads(4)
cd=torch.load(DATA/"original_completo_cd_v1.pt",map_location="cpu",weights_only=False)
model=M.TravelTimeModel(dim=64,hidden=128,dropout=.1,bank_size=max(cd.n_cells//2,8),pace=True,n_extra=len(M.POSENC_NAMES))
t=M.Trainer(model,cd,[],device="cuda",batch_size=1024,val_size=100)
t.sched=torch.optim.lr_scheduler.OneCycleLR(t.opt,max_lr=.001,total_steps=100,pct_start=.05,cycle_momentum=False)
batch=next(iter(t.tgt_loader))
timings={}
for name in ("original","cached_state"):
    if name=="cached_state":
        with torch.no_grad():
            t.model.eval();cached=[]
            for fold in range(3):
                hw=torch.arange(168,device="cuda")
                inp=M.S.assemble_st_input(cd.st.feat,cd.st.spd[fold],cd.st.obs[fold],t._win,hw)
                cached.append(t.model.stgnn(inp,cd.st.P_fwd,cd.st.P_bwd).detach())
            cached=torch.stack(cached)
        def frozen_window(self,city,idx,o,d,out_of_fold):
            how,fold=self._st_keys(city,idx,out_of_fold)
            return torch.cat([cached[fold,how,o],cached[fold,how,d]],1)
        t._window_embed=types.MethodType(frozen_window,t)
    values=[];t.model.train()
    for i in range(3):
        torch.cuda.synchronize();start=time.time();loss,ex=t._step(batch,[])
        torch.cuda.synchronize();values.append(time.time()-start)
    timings[name]=values;print(name,values,flush=True)
write_json(BASE/"instrumentos/perfil_original_completo.json",dict(seconds_per_step=timings,batch=1024,cells=cd.n_cells,full_epoch_steps=int(np.ceil(len(cd.train_idx)/1024)),caveat="A separate original training process may be concurrent; comparative order and GPU contention affect timings. Diagnostic steps only; cached state used current random/initial weights, not a fitted model."))

