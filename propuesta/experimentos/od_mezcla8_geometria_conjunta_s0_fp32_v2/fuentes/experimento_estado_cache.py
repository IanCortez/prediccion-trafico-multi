from common import *
from features import *
import torch,sys,argparse
sys.path.insert(0,str(ROOT/"codigo-ian/src"))
import tte_model as M
from torch.nn import functional as F
class CachedTrainer(M.Trainer):
    def __init__(self,*args,seconds_loss=False,no_invariance=True,**kwargs):
        super().__init__(*args,**kwargs);self.seconds_loss=seconds_loss;self.no_invariance=no_invariance
        self.counter=0;self.steps_per_full_epoch=int(np.ceil(len(self.target.train_idx)/self.batch_size))
        self.truth_gpu=self.target.raw_times.to(self.device)
    @torch.no_grad()
    def refresh(self,cd):
        was=self.model.stgnn.training;self.model.stgnn.eval();cache=[]
        for f in range(cd.st.spd.shape[0]):
            hour=torch.arange(168,device=self.device)
            H=M.S.assemble_st_input(cd.st.feat,cd.st.spd[f],cd.st.obs[f],self._win,hour)
            cache.append(self.model.stgnn(H,cd.st.P_fwd,cd.st.P_bwd))
        self.state_cache=torch.stack(cache).detach();self.model.stgnn.train(was)
    def _window_embed(self,cd,idx,o,d,out_of_fold):
        hour,fold=self._st_keys(cd,idx,out_of_fold)
        return torch.cat([self.state_cache[fold,hour,o],self.state_cache[fold,hour,d]],1)
    def _step(self,tb,sb):
        if self.counter%self.steps_per_full_epoch==0:self.refresh(self.target)
        out=super()._step(tb,sb);self.counter+=1
        if self.counter%250==0:
            print(dict(step=self.counter,epoch_fraction=self.counter/self.steps_per_full_epoch),flush=True)
        return out
    def predict(self,cd,idx):
        self.refresh(cd)
        return super().predict(cd,idx)
    def _loss_on(self,cd,batch):
        if not self.seconds_loss and not self.no_invariance:return super()._loss_on(cd,batch)
        feats,o,d,y,idx=(x.to(self.device) for x in batch)
        h=self.model.encode_cells(cd);state=self._window_embed(cd,idx,o,d,True)
        phi,psi=self.model.represent(h,o,d,feats,state);prediction=self.model.predict(phi,psi,feats)
        per=(torch.exp(prediction*cd.time_log_std+cd.time_log_mean)-self.truth_gpu[idx]).abs()/600 if self.seconds_loss else F.huber_loss(prediction,y,reduction="none",delta=1)
        loss=per.mean();extras={}
        if self.lambda_aux>0:
            target=cd.route_circuity[idx];valid=torch.isfinite(target)
            if valid.any():loss+=self.lambda_aux*F.huber_loss(self.model.aux_circuity(psi[valid]).squeeze(1),target[valid],delta=.5)
        if not self.no_invariance:
            env=self.model.env_infer(phi,psi).detach()
            loss+=self.alpha*self._concept_risk(env,per).var(unbiased=False)+self.beta*self._contrastive(phi,psi,env)
            extras=dict(phi=phi.detach(),psi=psi.detach(),per=per.detach())
        li=self._interp_loss(cd);extras["interp"]=li.item()
        return loss+self.lambda_st*li,extras
p=argparse.ArgumentParser();p.add_argument("--id",required=True);p.add_argument("--epochs",type=int,default=40);p.add_argument("--seed",type=int,default=0);p.add_argument("--huber",action="store_true");p.add_argument("--invariance",action="store_true");p.add_argument("--gps-state",action="store_true")
a=p.parse_args();run=RUNS/a.id;run.mkdir(exist_ok=False);snapshot(run,vars(a));started=time.time()
torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed)
file=DATA/("original_completo_gps_state_v1.pt" if a.gps_state else "original_completo_cd_v1.pt")
cd=torch.load(file,map_location="cpu",weights_only=False)
model=M.TravelTimeModel(dim=64,hidden=128,dropout=.1,bank_size=max(cd.n_cells//2,8),pace=True,n_extra=len(M.POSENC_NAMES))
trainer=CachedTrainer(model,cd,[],device="cuda",batch_size=1024,val_size=len(cd.val_idx),seconds_loss=not a.huber,no_invariance=not a.invariance,alpha=.5 if a.invariance else 0,beta=.5 if a.invariance else 0)
trainer.train(epochs=a.epochs,patience=10)
pred=trainer.predict(cd,cd.val_idx);truth=cd.raw_times[cd.val_idx].numpy();trainer.save(run/"modelo.pt")
x=np.load(DATA/"validation_x.npz")
np.savez_compressed(run/"validacion.npz",row=x["row"],timestamp=x["timestamp"],pred=pred,true=truth)
rec=dict(id=a.id,architecture="upstream cached-state",config=vars(a),validation=metrics(truth,pred),interval=block_interval(truth,pred,x["timestamp"]),train_n=len(cd.train_idx),epochs=len(trainer.history["train"]),elapsed_seconds=time.time()-started,model_sha256=sha(run/"modelo.pt"),city_cache_sha256=sha(file),data_manifest_sha256=sha(DATA/"manifiesto.json"),conclusion="Estado semanal precalculado por época. ST se entrena con interpolación; TTE usa estado desacoplado. Se refresca antes de validar y después de restaurar el mejor modelo. Comparar con ablaciones de pérdida/invariancia y velocidad local.")
write_json(run/"resultado.json",rec);register(rec);print(json.dumps(rec),flush=True)

