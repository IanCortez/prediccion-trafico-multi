"""Strict same-fold cross-fitted residual memory. All-bank exact Validation inference."""
from base_replanteo import *
from modelo_memoria_residual import ResidualMemory,stream_value
from inferencia import feature_input
import torch,random,argparse,shutil
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--id',required=True);ap.add_argument('--epochs',type=int,default=70);ap.add_argument('--seed',type=int,default=0);a=ap.parse_args();run=OUT/a.id;assert not run.exists();run.mkdir();(run/'fuentes').mkdir();start=time.time();access=deny_test();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
 cfg=dict(id=a.id,seed=a.seed,features=206,loss='MAE on base prediction plus gated learned soft-neighbor residual correction',base='4-fold UTC departure-week-blocked fixed1800-tree CatBoost; no held-fold-label fitting or early stopping',candidate_rule='Training queries and candidates belong to SAME heldout fold; their base excludes the entire query fold; own row always masked',bank='OOF residual true-minus-CFbase; all Train bank for Validation, base uses full Train',validation='Fixed4096 selection rows, final all127941, exact all-bank streaming values',query_batch=512,stochastic_candidates=16384,lr=.001,epochs=a.epochs,patience=12,Test_access=False,max_seconds=7200)
 write(run/'protocolo.json',cfg)
 for name in ['entrenar_residual.py','modelo_memoria_residual.py','base_replanteo.py']:shutil.copy2(OUT/name,run/'fuentes'/name)
 fcfg=dict(routes=True,context=True,projected_routes=True,mapmatched=False,calendar=True);tr,X,names=feature_input('train',dict(config=fcfg,architecture='residual_memory'));va,V,vnames=feature_input('validation',dict(config=fcfg,architecture='residual_memory'));assert names==vnames and len(names)==206;all_y=np.load(DATA/'train_y.npy');v=np.load(DATA/'validation_y.npy');eligible=(tr['timestamp']+all_y)<va['timestamp'].min();oof=np.load(OUT/'base_oof_estatica_s0/train_oof.npz');base_val=np.load(OUT/'base_oof_estatica_s0/validacion.npz');assert np.array_equal(oof['row'],tr['row'][eligible]);assert np.array_equal(base_val['row'],va['row']);assert np.array_equal(oof['true'],all_y[eligible]);assert np.array_equal(base_val['true'],v)
 X=X[eligible];y=oof['true'];base=oof['pred_oof'];fold=oof['fold'];geo=X[:,:4].copy();vgeo=V[:,:4].copy();mean=X.mean(0,dtype=np.float64).astype(np.float32);std=np.maximum(X.std(0,dtype=np.float64),1e-5).astype(np.float32);X=np.clip((X-mean)/std,-20,20);V=np.clip((V-mean)/std,-20,20);x=torch.tensor(X,device='cuda');vx=torch.tensor(V,device='cuda');g=torch.tensor(geo,device='cuda');vg=torch.tensor(vgeo,device='cuda');base_t=torch.tensor(base/600,device='cuda');values=torch.tensor((y-base)/600,device='cuda');target=torch.tensor(y/600,device='cuda');base_v=torch.tensor(base_val['pred']/600,device='cuda');groups=[torch.tensor(np.flatnonzero(fold==f),device='cuda') for f in range(4)];model=ResidualMemory(206).cuda();del X,V,geo,vgeo
 proxy=np.sort(np.random.default_rng(1729).choice(len(v),4096,replace=False));np.save(run/'selection_indices.npy',proxy);old=np.load(BEST/'validacion.npz');write(run/'baseline_selection.json',dict(n=len(proxy),retained=metric(v[proxy],old['pred'][proxy]),full_base=metric(v[proxy],base_val['pred'][proxy])))
 @torch.no_grad()
 def predict(indices=None):
  model.eval();keys=torch.empty((len(x),36),device='cuda')
  for first in range(0,len(x),4096):keys[first:first+4096]=model.encode(x[first:first+4096],g[first:first+4096])[0]
  use=np.arange(len(v)) if indices is None else indices;out=[]
  for first in range(0,len(use),512):
   ix=torch.tensor(use[first:first+512],device='cuda');n=len(ix)
   if n<512:ix=torch.cat([ix,ix[-1:].expand(512-n)])
   q,gate=model.encode(vx[ix],vg[ix]);corr=stream_value(q,keys,values);p=(base_v[ix]+.99*gate*corr).clamp(.1,2.975)*600;out.append(p[:n].cpu().numpy())
  return np.concatenate(out)
 # Own-label intervention plus exact candidate fold check before training.
 group=groups[0];qi=group[:1];ci=group[:64];model.eval();pred,w=model.correction(x[qi],g[qi],x[ci],g[ci],qi,ci,values[ci]);assert w[0,0]==0;vv=values[ci].clone();vv[0]+=999;changed,_=model.correction(x[qi],g[qi],x[ci],g[ci],qi,ci,vv);assert torch.equal(pred,changed);assert np.all(fold[ci.cpu().numpy()]==fold[qi.cpu().numpy()][0]);write(run/'verificacion_inicial.json',dict(passed=True,self_value_intervention_invariant=True,same_fold_candidates=True,base_source_sha256=sha(OUT/'base_oof_estatica_s0/train_oof.npz'),no_Test=True))
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=3,min_lr=1e-5);history=[];best=float('inf');bad=0
 for epoch in range(1,a.epochs+1):
  model.train();total=0;nseen=0
  for f in np.random.permutation(4):
   group=groups[f]
   for ix in group[torch.randperm(len(group),device='cuda')].split(512):
    ci=group[torch.randint(len(group),(cfg['stochastic_candidates'],),device='cuda')];opt.zero_grad(set_to_none=True);corr,_=model.correction(x[ix],g[ix],x[ci],g[ci],ix,ci,values[ci]);p=(base_t[ix]+corr).clamp(.1,2.975);loss=(p-target[ix]).abs().mean();assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=float(loss.detach())*len(ix);nseen+=len(ix)
  pp=predict(proxy);score=metric(v[proxy],pp);scheduler.step(score['mae_s']);r=dict(epoch=epoch,train_mae_s=600*total/nseen,selection_validation=score,tau=float(model.log_tau.exp().detach()),geo_bandwidth=model.log_geo_bw.exp().detach().cpu().numpy(),seconds=time.time()-start);history.append(r);write(run/'historia.json',history);print(json.dumps(r,default=lambda z:z.tolist()),flush=True)
  if score['mae_s']<best-.05:best=score['mae_s'];bad=0;best_epoch=epoch;torch.save(dict(state=model.state_dict(),mean=mean,std=std,features=names,config=cfg,base_model_sha256=sha(OUT/'base_oof_estatica_s0/modelo.cbm')),run/'modelo.pt')
  else:bad+=1
  if bad>=12 or time.time()-start>cfg['max_seconds']:break
 ck=torch.load(run/'modelo.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['state']);pp=predict();np.savez_compressed(run/'validacion.npz',row=va['row'],timestamp=va['timestamp'],true=v,pred=pp,base=base_val['pred']);result=dict(id=a.id,config=cfg,validation=metric(v,pp),base_validation=metric(v,base_val['pred']),best_epoch=best_epoch,epochs_run=len(history),train_n=len(y),model_sha256=sha(run/'modelo.pt'),code_sha256=sha(__file__),model_code_sha256=sha(OUT/'modelo_memoria_residual.py'),paired_vs_retained=paired(v,pp,old['pred'],va['timestamp']),paired_vs_base=paired(v,pp,base_val['pred'],va['timestamp']),actual_data_files_accessed=sorted(access),seconds=time.time()-start,Test_access=False,global_goal_validation=metric(v,pp)['mae_s']<73.5);write(run/'resultado.json',result);print(json.dumps({k:result[k] for k in ['id','validation','base_validation','best_epoch','seconds']}),flush=True)
if __name__=='__main__':main()

