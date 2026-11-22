"""End-to-end learned OD memory; no frozen TabM, no fixed neighbor list, no Test."""
from base_replanteo import *
from modelo_memoria import LearnedMemory,stream_distribution
from inferencia import feature_input
import torch,random,argparse,math
from torch import nn

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--id',required=True);ap.add_argument('--loss',choices=['cdf','mse'],default='cdf');ap.add_argument('--epochs',type=int,default=80);ap.add_argument('--seed',type=int,default=0);a=ap.parse_args();run=(OUT/a.id).resolve();assert run.parent==OUT and not run.exists();run.mkdir();start=time.time();access=deny_test();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
 cfg=dict(id=a.id,seed=a.seed,objective=a.loss,encoder=[206,256,128,32],query_batch=512,stochastic_candidates=16384,epochs=a.epochs,patience=12,lr=.001,validation_selection='8192 fixed Validation rows, exact entire training memory; final all127941',inference='Exact all-bank normalized kernel via streaming; no topK approximation',memory_features='206 static OD/map/calendar; no GPS label-derived profiles',self_exclusion=True,memory_arrival_embargo=True,query_duration_as_input=False,Test_access=False,source='ModernNCA ICLR2025 inspiration; custom distributional likelihood and geographic kernel, not exact paper replica',source_url='https://proceedings.iclr.cc/paper_files/paper/2025/file/ecd294194d5155432aab533f7ce0f6fe-Paper-Conference.pdf',max_seconds=7200)
 write(run/'protocolo.json',cfg)
 fcfg=dict(routes=True,context=True,projected_routes=True,mapmatched=False,calendar=True);tr,X,names=feature_input('train',dict(config=fcfg,architecture='learned_memory'));va,V,vnames=feature_input('validation',dict(config=fcfg,architecture='learned_memory'));assert names==vnames and len(names)==206 and not any(n.startswith('mm_') for n in names);y=np.load(DATA/'train_y.npy');v=np.load(DATA/'validation_y.npy');fit=(tr['timestamp']+y)<va['timestamp'].min();X=X[fit];y=y[fit];row=tr['row'][fit];assert len(np.unique(row))==len(row);geo=X[:,:4].copy();vgeo=V[:,:4].copy();mean=X.mean(0,dtype=np.float64).astype(np.float32);std=np.maximum(X.std(0,dtype=np.float64),1e-5).astype(np.float32);X=np.clip((X-mean)/std,-20,20);V=np.clip((V-mean)/std,-20,20);x=torch.tensor(X,device='cuda');vx=torch.tensor(V,device='cuda');g=torch.tensor(geo,device='cuda');vg=torch.tensor(vgeo,device='cuda');target=torch.tensor(y/600,device='cuda');label=torch.tensor(((y-60)/15).astype(np.int64),device='cuda');onehot=torch.zeros((len(y),116),device='cuda');onehot.scatter_(1,label[:,None],1.);prior=onehot.mean(0);target_mean=target.mean();grid=torch.arange(60,1786,15,device='cuda',dtype=torch.float32);model=LearnedMemory(206).cuda();del X,V,geo,vgeo
 # Independent streaming reference and exclusion of the query's own value.
 model.eval();tq=torch.randn(7,36,device='cuda');tk=torch.randn(150,36,device='cuda');tl=torch.arange(150,device='cuda')%116;th=nn.functional.one_hot(tl,116).float();tp=th.mean(0);ref=.99*(model.scores(tq,tk).softmax(-1)@th)+.01*tp;check=stream_distribution(tq,tk,th,tp,chunk=37);assert (ref-check).abs().max()<2e-6
 qid=torch.tensor([0],device='cuda');cid=torch.arange(64,device='cuda');w=model.neighborhood(x[:1],g[:1],x[:64],g[:64],qid,cid);assert w[0,0]==0 and torch.isfinite(w).all();own=onehot[:64].clone();pred=w@own;own[0]=own[0].roll(10);assert torch.equal(pred,w@own);write(run/'verificacion_inicial.json',dict(passed=True,streaming_max_error=float((ref-check).abs().max()),self_value_intervention_invariant=True,no_validation_labels_in_memory=True))
 proxy=np.sort(np.random.default_rng(1729).choice(len(v),8192,replace=False));np.save(run/'selection_indices.npy',proxy);proxy_gpu=torch.tensor(proxy,device='cuda');opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=3,min_lr=1e-5)
 @torch.no_grad()
 def predict(indices=None):
  model.eval();keys=torch.empty((len(x),36),device='cuda')
  for first in range(0,len(x),4096):keys[first:first+4096]=model.pack(x[first:first+4096],g[first:first+4096])
  qx=vx if indices is None else vx[indices];qg=vg if indices is None else vg[indices];out=[]
  for first in range(0,len(qx),512):
   xx=qx[first:first+512];gg=qg[first:first+512];n=len(xx)
   if n<512:xx=torch.cat([xx,xx[-1:].expand(512-n,-1)]);gg=torch.cat([gg,gg[-1:].expand(512-n,-1)])
   p=stream_distribution(model.pack(xx,gg),keys,onehot,prior)
   prediction=(p*grid).sum(-1) if a.loss=='mse' else grid[(p.cumsum(-1)<.5).sum(-1).clamp_max(115)];out.append(prediction[:n].cpu().numpy())
  return np.concatenate(out)
 best=float('inf');best_epoch=0;bad=0;history=[]
 for epoch in range(1,a.epochs+1):
  model.train();total=0
  for ix in torch.randperm(len(x),device='cuda').split(512):
   ci=torch.randint(len(x),(cfg['stochastic_candidates'],),device='cuda');opt.zero_grad(set_to_none=True);w=model.neighborhood(x[ix],g[ix],x[ci],g[ci],ix,ci)
   if a.loss=='mse':prediction=.99*(w@target[ci])+.01*target_mean;loss=(prediction-target[ix]).square().mean()
   else:
    p=.99*(w@onehot[ci])+.01*prior;nll=-p.gather(-1,label[ix,None]).clamp_min(1e-9).log().mean();rps=(p.cumsum(-1)-(grid[None,:]>=(60+15*label[ix,None])).float()).square().mean();loss=nll+5*rps
   assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=float(loss.detach())*len(ix)
  pred=predict(proxy_gpu);m=metric(v[proxy],pred);scheduler.step(m['mae_s']);r=dict(epoch=epoch,train_loss=total/len(x),selection_validation=m,seconds=time.time()-start,tau=float(model.log_tau.exp().detach()),geo_bandwidth=model.log_geo_bw.exp().detach().cpu().numpy());history.append(r);write(run/'historia.json',history);print(json.dumps(r,default=lambda z:z.tolist() if isinstance(z,np.ndarray) else z),flush=True)
  if m['mae_s']<best-.05:
   best=m['mae_s'];bad=0;best_epoch=epoch;torch.save(dict(state=model.state_dict(),mean=mean,std=std,features=names,config=cfg,memory_row_sha256=hashlib.sha256(row.tobytes()).hexdigest()),run/'modelo.pt');write(run/'mejor.json',r)
  else:bad+=1
  if bad>=12 or time.time()-start>cfg['max_seconds']:break
 ck=torch.load(run/'modelo.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['state']);pred=predict();np.savez_compressed(run/'validacion.npz',row=va['row'],timestamp=va['timestamp'],true=v,pred=pred);old=np.load(BEST/'validacion.npz');result=dict(id=a.id,config=cfg,validation=metric(v,pred),train_n=len(y),best_epoch=best_epoch,epochs_run=len(history),parameters=sum(p.numel() for p in model.parameters()),model_sha256=sha(run/'modelo.pt'),code_sha256=sha(__file__),model_code_sha256=sha(OUT/'modelo_memoria.py'),manifest_sha256=sha(DATA/'manifiesto.json'),paired_vs_retained=paired(v,pred,old['pred'],va['timestamp']),actual_data_files_accessed=sorted(access),seconds=time.time()-start,global_goal_validation=metric(v,pred)['mae_s']<73.5,Test_access=False)
 write(run/'resultado.json',result);print(json.dumps({k:result[k] for k in ['id','validation','best_epoch','epochs_run','seconds']}),flush=True)
if __name__=='__main__':main()

