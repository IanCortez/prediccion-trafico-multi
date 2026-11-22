"""Fresh joint route/time model with global historical-route prior; original cohort unchanged."""
from piloto_transferencia import ROOT,BASE,OUT,FROZEN,write,sha,paired
from adaptar_rutas import metric
from inferencia import feature_input
from modelo_mezcla_conjunta import JointRouteTimeMixture
import numpy as np,torch,random,time,json,os,sys
import rtdl_num_embeddings as rtdl
BEST=BASE/'experimentos/od_mezcla8_geometria_conjunta_s0_fp32_v2'

def main():
 start=time.time();run=OUT/'modelo_global_conjunto_s0';run.mkdir(exist_ok=False);torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0);random.seed(0);torch.set_float32_matmul_precision('high');accessed=[]
 def audit(event,args):
  if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
   try:p=ROOT.__class__(os.fsdecode(args[0])).resolve()
   except (OSError,ValueError):return
   if p.is_relative_to(BASE/'datos'):
    assert not p.name.startswith('test'),'Test is sealed'
    if str(p) not in accessed:accessed.append(str(p))
 sys.addaudithook(audit)
 rec=json.loads((BEST/'resultado.json').read_text(encoding='utf-8-sig'));cfg=dict(seed=0,epochs=80,batch=512,k=16,width=512,geometry_weight=.2,lr=.001,weight_decay=.0003,patience=15,max_seconds=3600,route_generation='global historical GPS time * (1 - .6 link_support), A-star',anchors=['mm_length_gps_time_600s','mm_free_gps_time_600s','global_history_600s','od_km'],geometry_priors=['projected_length','projected_free','global','none'],loss='original NLL +5RPS +.2 conditional geometry score',validation_selection_precision='bfloat16; final candidate replay float32 canonical2048',global_target_mae_s=73.5,Test_access=False,query_actual_route_input=False)
 write(run/'protocolo.json',cfg)
 tr,X,names=feature_input('train',rec);va,V,vnames=feature_input('validation',rec);z=np.load(OUT/'rutas_globales_completas/train_rutas_globales.npz');zv=np.load(OUT/'rutas_globales_completas/validation_rutas_globales.npz');assert np.array_equal(z['row'],tr['row']) and np.array_equal(zv['row'],va['row']);X=np.column_stack([X,z['X']]);V=np.column_stack([V,zv['X']]);names+=z['names'].tolist();y=np.load(BASE/'datos/train_y.npy');v=np.load(BASE/'datos/validation_y.npy');keep=np.flatnonzero(np.ptp(X,axis=0)>1e-7);mean=X[:,keep].mean(0,dtype=np.float64).astype(np.float32);std=np.maximum(X[:,keep].std(0,dtype=np.float64),1e-5).astype(np.float32);normalized=(X[:,keep]-mean)/std;ix=np.random.default_rng(0).choice(len(y),min(200000,len(y)),replace=False);bins=rtdl.compute_bins(torch.as_tensor(np.vstack([normalized[ix],normalized.min(0),normalized.max(0)])),n_bins=48);del normalized
 anchor_indices=[names.index(n) for n in cfg['anchors']];anchors=[]
 for col,scale in zip(anchor_indices,[600,600,600,1000]):
  logx=np.log(np.maximum(X[:,col].astype(float)*scale,.1));beta,intercept=np.polyfit(logx,np.log(y),1);intercept+=np.log(np.median(y/np.exp(intercept+beta*logx)));anchors.append([intercept,beta])
 short=[names.index('projected_length_shape_'+str(i)) for i in range(14)];fast=[names.index('projected_free_shape_'+str(i)) for i in range(14)];glob=[names.index('global_shape_'+str(i)) for i in range(14)];geometry_indices=np.array([short,fast,glob,[-1]*14]*2);model=JointRouteTimeMixture(mean,std,keep,bins,anchors,anchor_indices,geometry_indices,16,512).cuda();gt=dict(np.load(BASE/'datos/geometria_train.npz'));assert np.all(np.diff(gt['row'])>0);pos=np.searchsorted(gt['row'],tr['row']);safe=np.minimum(pos,len(gt['row'])-1);valid=(pos<len(gt['row']))&(gt['row'][safe]==tr['row'])&(gt['X'][safe,-1]<=500);geometry=gt['X'][safe,1:15].copy();valid&=np.isfinite(geometry).all(1);geometry[~valid]=0
 x=torch.tensor(X,device='cuda');vx=torch.tensor(V,device='cuda');target=torch.tensor(y,device='cuda');g=torch.tensor(geometry,device='cuda');mask=torch.tensor(valid,device='cuda',dtype=torch.float32);del X,V,geometry;opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=5,min_lr=1e-5)
 @torch.no_grad()
 def predict(fp32=False):
  model.eval();out=[];torch.set_float32_matmul_precision('highest' if fp32 else 'high')
  for block in vx.split(2048):
   n=len(block)
   if n<2048:block=torch.cat([block,block[-1:].expand(2048-n,-1)])
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=not fp32):value=model(block)
   out.append(model.quantiles(value[0],(.5,))[:n,0].cpu().numpy())
  torch.set_float32_matmul_precision('high');return np.concatenate(out)
 history=[];best=float('inf');bad=0;best_epoch=0
 for epoch in range(1,cfg['epochs']+1):
  model.train();total=0
  for ix in torch.randperm(len(x),device='cuda').split(cfg['batch']):
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):output=model(x[ix])
   logp=output[0];label=((target[ix]-60)/15).round().long();nll=-logp.gather(-1,label[:,None,None].expand(-1,16,1)).mean();desired=(model.duration_grid[None,None,:]>=target[ix,None,None]).float();rps=(logp.exp().cumsum(-1)-desired).square().mean();loss=nll+5*rps+.2*model.geometry_score(output,target[ix],g[ix],mask[ix]);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=float(loss.detach())*len(ix)
  p=predict();m=metric(v,p);scheduler.step(m['mae_s']);r=dict(epoch=epoch,train_loss=total/len(x),validation=m,seconds=time.time()-start,lr=opt.param_groups[0]['lr']);history.append(r);write(run/'historia.json',history);print(json.dumps(r),flush=True)
  if m['mae_s']<best-.02:
   best=m['mae_s'];best_epoch=epoch;bad=0;torch.save(dict(state=model.state_dict(),mean=mean,std=std,keep=keep,bins=[b.cpu() for b in bins],anchors=anchors,anchor_indices=anchor_indices,geometry_indices=geometry_indices,config=cfg,features=names),run/'modelo.pt');write(run/'mejor.json',r)
  else:bad+=1
  if bad>=cfg['patience'] or time.time()-start>cfg['max_seconds']:break
 ck=torch.load(run/'modelo.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['state']);p=predict(True);old=np.load(BEST/'validacion.npz');np.savez_compressed(run/'validacion.npz',row=va['row'],timestamp=va['timestamp'],true=v,pred=p);result=dict(id=run.name,config=cfg,features=names,validation=metric(v,p),best_epoch=best_epoch,epochs_run=len(history),parameters=sum(p.numel() for p in model.parameters()),model_sha256=sha(run/'modelo.pt'),code_sha256=sha(ROOT.__class__(__file__)),data_manifest_sha256=sha(BASE/'datos/manifiesto.json'),paired_vs_retained=paired(v,p,old['pred'],va['timestamp']),seconds=time.time()-start,actual_data_files_accessed=accessed,Test_access=False,global_goal_validation=metric(v,p)['mae_s']<73.5,conclusion='Fresh joint model: same original objective, additional global OD-only route and prior. Comparison is exploratory on reused Validation.')
 write(run/'resultado.json',result);print(json.dumps(result,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else str(v)),flush=True)
if __name__=='__main__':main()
