"""MAE-aligned fine tuning of retained OD model. No Test or new query inputs."""
from piloto_transferencia import ROOT,BASE,OUT,FROZEN,write,sha,metric,paired
import sys,os,time,json,random,argparse
import numpy as np
import torch
from inferencia import feature_input
from modelo_mezcla_rutas import make_duration_model
BEST=BASE/'experimentos/od_mezcla8_geometria_conjunta_s0_fp32_v2'

def median_continuous(prob,grid):
 cdf=prob.cumsum(-1);i=(cdf<.5).sum(-1).clamp_max(len(grid)-1)
 p=prob.gather(-1,i[...,None])[...,0].clamp_min(1e-7)
 c=cdf.gather(-1,i[...,None])[...,0]-p
 return (grid[i]-7.5+15*(.5-c)/p).clamp(60,1785)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=12);ap.add_argument('--seed',type=int,default=20261121);ap.add_argument('--id',default='mae_mediana_s0');a=ap.parse_args()
 run=(OUT/a.id).resolve();assert run.parent==OUT;run.mkdir(exist_ok=False);started=time.time();accessed=[]
 def audit(event,args):
  if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
   try:p=ROOT.__class__(os.fsdecode(args[0])).resolve()
   except (OSError,ValueError):return
   if p.is_relative_to(BASE/'datos'):
    assert not p.name.startswith('test'),'Test is sealed'
    if str(p) not in accessed:accessed.append(str(p))
 sys.addaudithook(audit)
 torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 rec=json.loads((BEST/'resultado.json').read_text(encoding='utf-8-sig'));assert sha(BEST/'modelo.pt')==rec['model_sha256']
 cfg=dict(seed=a.seed,epochs=a.epochs,lr=.0001,weight_decay=.0003,batch=512,patience=4,loss='0.5 MAE mixture median + 0.5 mean member MAE; divided by600; +0.02NLL +0.1RPS',initial_model_sha256=rec['model_sha256'],inference_precision='float32',median_mode='discrete',validation_selection='full 127941, exploratory reused validation',max_seconds=1800,max_additional_bytes=1000000000,query_information='unchanged OD/departure; no query trajectory or duration')
 write(run/'protocolo.json',cfg)
 tr,X,names=feature_input('train',rec);va,V,vnames=feature_input('validation',rec);assert names==vnames==rec['features'];y=np.load(BASE/'datos/train_y.npy');v=np.load(BASE/'datos/validation_y.npy');ck=torch.load(BEST/'modelo.pt',map_location='cpu',weights_only=False);model=make_duration_model(ck,rec['config']).cuda();model.load_state_dict(ck['state']);grid=model.duration_grid
 x=torch.tensor(X,device='cuda');target=torch.tensor(y,device='cuda');del X
 @torch.no_grad()
 def predict():
  model.eval();out=[];cont=[]
  for first in range(0,len(V),2048):
   z=torch.tensor(V[first:first+2048],device='cuda');n=len(z)
   if n<2048:z=torch.cat([z,z[-1:].expand(2048-n,-1)])
   logp=model(z)[0];prob=logp.exp().mean(1);out.append(grid[(prob.cumsum(-1)<.5).sum(-1).clamp_max(115)][:n].cpu().numpy());cont.append(median_continuous(prob,grid)[:n].cpu().numpy())
  return np.concatenate(out),np.concatenate(cont)
 original,_=predict();saved=np.load(BEST/'validacion.npz');assert np.array_equal(original,saved['pred']);best=metric(v,original)['mae_s'];best_epoch=0
 # Meaningful numerical checks: smooth median gradients and exact original inference.
 model.eval();z=model(x[:32])[0].exp();q=median_continuous(z,grid);assert torch.isfinite(q).all() and ((q>=60)&(q<=1785)).all();loss=(q-target[:32,None]).abs().mean()/600;loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);model.zero_grad(set_to_none=True)
 opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);history=[];bad=0
 for epoch in range(1,a.epochs+1):
  model.train();total=0.;count=0
  for ix in torch.randperm(len(x),device='cuda').split(cfg['batch']):
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):logp=model(x[ix])[0]
   prob=logp.exp();member=median_continuous(prob,grid);mixture=median_continuous(prob.mean(1),grid);t=target[ix];label=((t-60)/15).long();nll=-logp.gather(-1,label[:,None,None].expand(-1,16,1)).mean();desired=(grid[None,None,:]>=t[:,None,None]).float();rps=(prob.cumsum(-1)-desired).square().mean();l1=(.5*(mixture-t).abs().mean()+.5*(member-t[:,None]).abs().mean())/600;loss=l1+.02*nll+.1*rps
   assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();total+=float(loss.detach())*len(ix);count+=len(ix)
  p,pc=predict();m=metric(v,p);r=dict(epoch=epoch,train_loss=total/count,validation=m,continuous_validation=metric(v,pc),seconds=time.time()-started);history.append(r);write(run/'historia.json',history);print(json.dumps(r),flush=True)
  if m['mae_s']<best-.02:
   best=m['mae_s'];best_epoch=epoch;bad=0;ck['state']={k:value.detach().cpu() for k,value in model.state_dict().items()};ck['fine_tuning_config']=cfg;torch.save(ck,run/'modelo.pt');np.savez_compressed(run/'validacion.npz',row=va['row'],timestamp=va['timestamp'],true=v,pred=p,continuous_pred=pc);write(run/'mejor.json',r)
  else:bad+=1
  if bad>=cfg['patience'] or time.time()-started>cfg['max_seconds']:break
 p=np.load(run/'validacion.npz')['pred'] if (run/'validacion.npz').exists() else original
 result=dict(id=a.id,config=cfg,base_mae_s=metric(v,original)['mae_s'],validation=metric(v,p),best_epoch=best_epoch,epochs_run=len(history),paired=paired(v,p,original,va['timestamp']),seconds=time.time()-started,code_sha256=sha(ROOT.__class__(__file__)),data_files_accessed=accessed,model_sha256=sha(run/'modelo.pt') if (run/'modelo.pt').exists() else None,Test_access=False,global_goal_met_on_validation=best<73.5)
 write(run/'resultado.json',result);print(json.dumps(result,default=lambda z:z.tolist() if isinstance(z,np.ndarray) else str(z)),flush=True)
if __name__=='__main__':main()
