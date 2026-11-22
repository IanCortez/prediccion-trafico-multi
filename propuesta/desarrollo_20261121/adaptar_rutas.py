"""Matched frozen-base conditional distribution adapters with/without global routes."""
from piloto_transferencia import ROOT,BASE,OUT,FROZEN,write,sha,paired
from ajuste_mae import median_continuous
from inferencia import feature_input
from modelo_mezcla_rutas import make_duration_model
import numpy as np,torch,time,json,random,os,argparse
from torch import nn
BEST=BASE/'experimentos/od_mezcla8_geometria_conjunta_s0_fp32_v2'

def metric(y,p):
 y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64);assert np.isfinite(p).all()
 return dict(n=len(y),mae_s=float(np.abs(y-p).mean()),rmse_s=float(np.sqrt(np.mean((y-p)**2))),mdae_s=float(np.median(np.abs(y-p))))

class Tilt(nn.Module):
 def __init__(self,n):
  super().__init__();self.net=nn.Sequential(nn.Linear(n+116,128),nn.ReLU(),nn.Dropout(.05),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,116));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def forward(self,x,base):
  energy=1.5*torch.tanh(self.net(torch.cat([x,base],-1))/1.5);p=base*energy.exp();return p/p.sum(-1,keepdim=True)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=20261121);parser.add_argument('--epochs',type=int,default=20);a=parser.parse_args();start=time.time();run=OUT/'adaptacion_rutas';run.mkdir(exist_ok=False);torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed);accessed=[]
 def audit(event,args):
  if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
   try:p=ROOT.__class__(os.fsdecode(args[0])).resolve()
   except (OSError,ValueError):return
   if p.is_relative_to(BASE/'datos'):
    assert not p.name.startswith('test'),'Test is sealed'
    if str(p) not in accessed:accessed.append(str(p))
 torch.set_num_threads(4);import sys;sys.addaudithook(audit)
 rec=json.loads((BEST/'resultado.json').read_text(encoding='utf-8-sig'));assert sha(BEST/'modelo.pt')==rec['model_sha256']
 cfg=dict(seed=a.seed,max_epochs=a.epochs,patience=5,batch=4096,lr=.001,loss='MAE continuous mixture median /600 + .02NLL +.1RPS + .1 KL to frozen base',tilt_bound=1.5,matched_controls=['same adapter with global-route inputs masked to zero','same adapter with global-route inputs'],base_frozen=True,base_model_sha256=rec['model_sha256'],new_route_features=21,storage='One mean CDF per Train query kept in GPU memory only; no teacher files or banks',Test_access=False,max_training_seconds=1800)
 write(run/'protocolo.json',cfg)
 tr,X,names=feature_input('train',rec);va,V,vnames=feature_input('validation',rec);assert names==vnames==rec['features'];y=np.load(BASE/'datos/train_y.npy');v=np.load(BASE/'datos/validation_y.npy')
 ck=torch.load(BEST/'modelo.pt',map_location='cpu',weights_only=False);base=make_duration_model(ck,rec['config']).cuda();base.load_state_dict(ck['state']);base.eval();base.requires_grad_(False);grid=base.duration_grid.detach().clone()
 @torch.no_grad()
 def teacher(X):
  out=torch.empty((len(X),116),device='cuda')
  for first in range(0,len(X),2048):
   x=torch.tensor(X[first:first+2048],device='cuda');n=len(x)
   if n<2048:x=torch.cat([x,x[-1:].expand(2048-n,-1)],0)
   out[first:first+n]=base(x)[0].exp().mean(1)[:n]
  return out
 pt=teacher(X);pv=teacher(V);original=grid[(pv.cumsum(-1)<.5).sum(-1).clamp_max(115)].cpu().numpy();assert np.array_equal(original,np.load(BEST/'validacion.npz')['pred']);print('Teacher mean CDF generated in RAM; exact Validation replay.',flush=True);del base,ck
 z=np.load(OUT/'rutas_globales_completas/train_rutas_globales.npz');zv=np.load(OUT/'rutas_globales_completas/validation_rutas_globales.npz');assert np.array_equal(tr['row'],z['row']) and np.array_equal(va['row'],zv['row']);route_names=z['names'].tolist();allnames=names+route_names;X=np.column_stack([X,z['X']]);V=np.column_stack([V,zv['X']]);mean=X.mean(0,dtype=np.float64).astype(np.float32);std=np.maximum(X.std(0,dtype=np.float64),1e-5).astype(np.float32);x=torch.tensor(np.clip((X-mean)/std,-20,20),device='cuda');vx=torch.tensor(np.clip((V-mean)/std,-20,20),device='cuda');del X,V;target=torch.tensor(y,device='cuda');label=((target-60)/15).long();allresults={}
 for variant in ['control_capacidad','rutas_globales']:
  torch.manual_seed(a.seed);model=Tilt(x.shape[1]).cuda();opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=.001);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=2,min_lr=1e-5);best=metric(v,original)['mae_s'];best_epoch=0;bad=0;history=[];vrun=run/variant;vrun.mkdir();chosen=original.copy();with_routes=variant=='rutas_globales'
  def head_input(data):
   if with_routes:return data
   data=data.clone();data[:,-21:]=0;return data
  @torch.no_grad()
  def predict():
   model.eval();out=[]
   for i in range(0,len(vx),2048):
    xx=vx[i:i+2048];bb=pv[i:i+2048];n=len(xx)
    if n<2048:xx=torch.cat([xx,xx[-1:].expand(2048-n,-1)]);bb=torch.cat([bb,bb[-1:].expand(2048-n,-1)])
    prob=model(head_input(xx),bb);out.append(grid[(prob.cumsum(-1)<.5).sum(-1).clamp_max(115)][:n].cpu().numpy())
   return np.concatenate(out)
  assert np.array_equal(predict(),original)
  # Numerical contract checks before the fit.
  model.eval();pr=model(head_input(x[:32]),pt[:32]);assert torch.isfinite(pr).all() and torch.max((pr.sum(-1)-1).abs())<1e-5;loss=median_continuous(pr,grid).mean();loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);model.zero_grad(set_to_none=True)
  for epoch in range(1,a.epochs+1):
   model.train();total=0
   for ix in torch.randperm(len(x),device='cuda').split(cfg['batch']):
    opt.zero_grad(set_to_none=True);prob=model(head_input(x[ix]),pt[ix]);logp=prob.clamp_min(1e-8).log();nll=-logp.gather(-1,label[ix,None]).mean();rps=(prob.cumsum(-1)-(grid[None,:]>=target[ix,None]).float()).square().mean();kl=(prob*(logp-pt[ix].clamp_min(1e-8).log())).sum(-1).mean();l1=(median_continuous(prob,grid)-target[ix]).abs().mean()/600;loss=l1+.02*nll+.1*rps+.1*kl;assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();total+=float(loss.detach())*len(ix)
   p=predict();m=metric(v,p);scheduler.step(m['mae_s']);r=dict(variant=variant,epoch=epoch,loss=total/len(x),validation=m,seconds=time.time()-start);history.append(r);write(vrun/'historia.json',history);print(json.dumps(r),flush=True)
   if m['mae_s']<best-.02:
    best=m['mae_s'];best_epoch=epoch;bad=0;chosen=p.copy();torch.save(dict(state=model.state_dict(),mean=mean,std=std,features=allnames,with_routes=with_routes,base_model_sha256=rec['model_sha256'],config=cfg),vrun/'modelo.pt');np.savez_compressed(vrun/'validacion.npz',row=va['row'],timestamp=va['timestamp'],true=v,pred=chosen)
   else:bad+=1
   if bad>=cfg['patience'] or time.time()-start>cfg['max_training_seconds']:break
  if not (vrun/'validacion.npz').exists():np.savez_compressed(vrun/'validacion.npz',row=va['row'],timestamp=va['timestamp'],true=v,pred=chosen)
  result=dict(variant=variant,validation=metric(v,chosen),best_epoch=best_epoch,epochs_run=len(history),model_sha256=sha(vrun/'modelo.pt') if (vrun/'modelo.pt').exists() else None,paired_vs_frozen=paired(v,chosen,original,va['timestamp']),Test_access=False);write(vrun/'resultado.json',result);allresults[variant]=result
 control=np.load(run/'control_capacidad/validacion.npz')['pred'];candidate=np.load(run/'rutas_globales/validacion.npz')['pred'];delta=paired(v,candidate,control,va['timestamp']);result=dict(config=cfg,results=allresults,paired_routes_vs_matched_control=delta,code_sha256=sha(ROOT.__class__(__file__)),data_files_accessed=accessed,seconds=time.time()-start,Test_access=False,goal_validation=metric(v,candidate)['mae_s']<73.5,conclusion='Exploratory repeated Validation; no Test certification unless candidate meets the target and is frozen first. Route geometry improvement alone is insufficient.')
 write(run/'resultado.json',result);print(json.dumps(result,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else str(v)),flush=True)
if __name__=='__main__':main()

