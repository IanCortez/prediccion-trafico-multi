"""Train the decomposed generative model. Query inputs remain strictly OD/departure."""
from base_replanteo import *
from inferencia import feature_input
from modelo_descompuesto import DecomposedTime
import torch,argparse,random,rtdl_num_embeddings as rtdl

def main():
 p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('--aux-weight',type=float,required=True,choices=[0.,1.]);p.add_argument('--epochs',type=int,default=100);p.add_argument('--seed',type=int,default=0);p.add_argument('--verify-only',action='store_true');a=p.parse_args();start=time.time();run=(OUT/a.id).resolve();assert run.parent==OUT and not run.exists();run.mkdir();access=deny_test();torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed);torch.set_float32_matmul_precision('high');rec=read(BEST/'resultado.json');cfg=dict(id=a.id,seed=a.seed,aux_weight=a.aux_weight,epochs=a.epochs,k=16,width=512,batch=512,lr=.001,patience=20,loss='NLL(total)+5RPS(total)+lambda NLL(low-count | total); total PMF is convolution of movement and low-displacement PMFs',input='Same 216 OD/departure/map/history features as retained model',support='Condition the joint mixture on 4 <= moving_n + low_n <=119 intervals',global_stop_prior='Fixed 50% point mass at zero + 50% geometric mean approx8; no count-derived initialization',anchor_fit='Total-time labels only, same for lambda0 and lambda1',Test_access=False,max_seconds=7200)
 write(run/'protocolo.json',cfg);tr,X,names=feature_input('train',rec);va,V,vnames=feature_input('validation',rec);assert names==vnames==rec['features'];y=np.load(DATA/'train_y.npy');v=np.load(DATA/'validation_y.npy');z=np.load(OUT/'descomposicion_train.npz');assert np.array_equal(z['row'],tr['row']);keep=np.flatnonzero(np.ptp(X,axis=0)>1e-7);mean=X[:,keep].mean(0,dtype=np.float64).astype(np.float32);std=np.maximum(X[:,keep].std(0,dtype=np.float64),1e-5).astype(np.float32);normalized=(X[:,keep]-mean)/std;ix=np.random.default_rng(a.seed).choice(len(y),min(200000,len(y)),replace=False);bins=rtdl.compute_bins(torch.as_tensor(np.vstack([normalized[ix],normalized.min(0),normalized.max(0)])),n_bins=48);del normalized
 anchor_names=['mm_length_gps_time_600s','mm_free_gps_time_600s','projected_free_freeflow_600s','od_km'];indices=[names.index(n) for n in anchor_names];anchors=[]
 for i,scale in zip(indices,[600,600,600,1000]):
  lx=np.log(np.maximum(X[:,i].astype(float)*scale,.1));b,c=np.polyfit(lx,np.log(y),1);c+=np.log(np.median(y/np.exp(c+b*lx)));anchors.append([c,b])
 model=DecomposedTime(mean,std,keep,bins,anchors,indices).cuda();x=torch.tensor(X,device='cuda');vx=torch.tensor(V,device='cuda');yn=torch.tensor(y/15,device='cuda');low=torch.tensor(z['low_n'].astype(np.float32),device='cuda');mask=torch.tensor(z['reliable'].astype(np.float32),device='cuda');target=torch.tensor(y,device='cuda');del X,V
 def lossfn(out,ix):
  logp=out[0];label=(yn[ix]-4).long();nll=-logp.gather(-1,label[:,None,None].expand(-1,16,1)).mean();desired=(model.duration_grid[None,None,:]>=target[ix,None,None]).float();rps=(logp.exp().cumsum(-1)-desired).square().mean();aux=model.conditional_count_score(out,yn[ix],low[ix],mask[ix]) if a.aux_weight else nll*0;return nll+5*rps+a.aux_weight*aux
 # Numerical contracts: finite gradients, normalized total PMFs and FFT vs independent direct convolution.
 model.eval();sample=torch.arange(16,device='cuda');out=model(x[sample]);assert float((out[0].exp().sum(-1)-1).abs().max())<1e-5;pm=out[1][0,0,0].exp().detach().cpu().numpy().astype(float);ps=out[2][0,0,0].exp().detach().cpu().numpy().astype(float);reference=np.convolve(pm,ps);fft=np.fft.irfft(np.fft.rfft(pm,256)*np.fft.rfft(ps,256),256)[:239];assert np.max(np.abs(reference-fft))<1e-10;refmix=sum(float(out[3][0,0,j].exp().detach().cpu())*np.convolve(out[1][0,0,j].exp().detach().cpu().numpy().astype(float),out[2][0,0,j].exp().detach().cpu().numpy().astype(float)) for j in range(4))[4:120];refmix/=refmix.sum();assert np.max(np.abs(refmix-out[0][0,0].exp().detach().cpu().numpy()))<1e-6;loss=lossfn(out,sample);assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);model.zero_grad(set_to_none=True)
 write(run/'verificacion_inicial.json',dict(passed=True,finite_gradients=True,total_probabilities_normalized=True,FFT_direct_convolution_error=float(np.max(np.abs(reference-fft))),low_count_is_not_query_input=True,Test_access=False))
 if a.verify_only:return
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0003);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=5,min_lr=1e-5)
 @torch.no_grad()
 def predict(fp32=False):
  model.eval();torch.set_float32_matmul_precision('highest' if fp32 else 'high');out=[]
  for block in vx.split(2048):
   n=len(block)
   if n<2048:block=torch.cat([block,block[-1:].expand(2048-n,-1)])
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=not fp32):p=model.median(model(block)[0])
   out.append(p[:n].cpu().numpy())
  torch.set_float32_matmul_precision('high');return np.concatenate(out)
 best=float('inf');bad=0;best_epoch=0;history=[]
 for epoch in range(1,a.epochs+1):
  model.train();total=0
  for ix in torch.randperm(len(x),device='cuda').split(512):
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):out=model(x[ix])
   loss=lossfn(out,ix);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=float(loss.detach())*len(ix)
  pred=predict();m=metric(v,pred);scheduler.step(m['mae_s']);r=dict(epoch=epoch,train_loss=total/len(x),validation=m,seconds=time.time()-start,lr=opt.param_groups[0]['lr']);history.append(r);write(run/'historia.json',history);print(json.dumps(r),flush=True)
  if m['mae_s']<best-.02:
   best=m['mae_s'];bad=0;best_epoch=epoch;torch.save(dict(state=model.state_dict(),mean=mean,std=std,keep=keep,bins=[b.cpu() for b in bins],anchors=anchors,anchor_indices=indices,features=names,config=cfg),run/'modelo.pt');write(run/'mejor.json',r)
  else:bad+=1
  if bad>=20 or time.time()-start>cfg['max_seconds']:break
 ck=torch.load(run/'modelo.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['state']);pred=predict(True);np.savez_compressed(run/'validacion.npz',row=va['row'],timestamp=va['timestamp'],true=v,pred=pred);old=np.load(BEST/'validacion.npz');result=dict(id=a.id,config=cfg,validation=metric(v,pred),best_epoch=best_epoch,epochs_run=len(history),model_sha256=sha(run/'modelo.pt'),code_sha256=sha(__file__),model_code_sha256=sha(OUT/'modelo_descompuesto.py'),counts_sha256=sha(OUT/'descomposicion_train.npz'),manifest_sha256=sha(DATA/'manifiesto.json'),paired_vs_retained=paired(v,pred,old['pred'],va['timestamp']),parameters=sum(p.numel() for p in model.parameters()),actual_data_files_accessed=sorted(access),seconds=time.time()-start,Test_access=False,global_target_on_validation=metric(v,pred)['mae_s']<73.5)
 write(run/'resultado.json',result);print(json.dumps({k:result[k] for k in ['id','validation','best_epoch','epochs_run','seconds']}),flush=True)
if __name__=='__main__':main()

