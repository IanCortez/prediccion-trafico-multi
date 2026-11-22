"""Train local spatial dynamics, then total-duration likelihood through exact route marginalization."""
from base_replanteo import *
from modelo_espacial import SpatialArrival
from inferencia import feature_input
import torch,random,argparse,shutil
from torch import nn


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--id',required=True);ap.add_argument('--epochs',type=int,default=30);ap.add_argument('--fine-steps',type=int,default=1000);ap.add_argument('--seed',type=int,default=0);a=ap.parse_args();run=(OUT/a.id).resolve();assert run.parent==OUT and not run.exists();run.mkdir();(run/'fuentes').mkdir();start=time.time();access=deny_test();torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed);torch.set_float32_matmul_precision('high')
 cfg=dict(id=a.id,seed=a.seed,epochs=a.epochs,fine_steps=a.fine_steps,local_batch=2048,dp_batch=4,inference_batch=32,selection_n=512,selection_head='fixed mixed spatial/direct posterior',local_loss='next-zone CE + residence CE + terminal CE + direct OD NLL +5RPS',fine_loss='mixed total NLL +5RPS through differentiable semi-Markov inference',query_input='Same216 OD/departure/map/history features, plus grid IDs derived deterministically from given endpoints',no_query_route_or_duration=True,geometry='200m cells; public2014 roads and Train transitions only',time_model='Spatial state transition + positive movement residence + nonnegative terminal residence; marginalize all paths',time_window='Fixed60..1785s for every query; no observed query point count',max_seconds=7200,Test_access=False)
 write(run/'protocolo.json',cfg)
 for name in ['entrenar_espacial.py','modelo_espacial.py','nucleo_semi_markov.py','base_replanteo.py']:shutil.copy2(OUT/name,run/'fuentes'/name)
 rec=read(BEST/'resultado.json');assert sha(BEST/'modelo.pt')==rec['model_sha256'];tr,X,names=feature_input('train',rec);va,V,vnames=feature_input('validation',rec);assert names==vnames==rec['features'];y=np.load(DATA/'train_y.npy');v=np.load(DATA/'validation_y.npy');stats=torch.load(BEST/'modelo.pt',map_location='cpu',weights_only=False);grid=read(OUT/'datos_semi_markov/protocolo.json');graph=dict(np.load(OUT/'datos_semi_markov/grafo_grid.npz'));distance=np.load(OUT/'datos_semi_markov/distancias_grid.npy');model=SpatialArrival(stats,grid,graph,distance,stats['anchors'][3]).cuda();model_stats={k:stats[k] for k in ['mean','std','keep','bins']};power=stats['anchors'][3];del stats
 def ids(d,tag):
  xx=(d[tag+'_lon']+8.62)*111195*np.cos(np.deg2rad(41.16));yy=(d[tag+'_lat']-41.16)*111195;aa=np.floor((xx-grid['x0'])/grid['cell_m']).astype(np.int64);bb=np.floor((yy-grid['y0'])/grid['cell_m']).astype(np.int64);assert np.all((aa>=0)&(aa<grid['nx'])&(bb>=0)&(bb<grid['ny']));return aa*grid['ny']+bb
 origin=torch.tensor(ids(tr,'o'),device='cuda');dest=torch.tensor(ids(tr,'d'),device='cuda');vo=torch.tensor(ids(va,'o'),device='cuda');vd=torch.tensor(ids(va,'d'),device='cuda');x=torch.tensor(X,device='cuda');vx=torch.tensor(V,device='cuda');target=torch.tensor(y,device='cuda');del X,V
 data=np.load(OUT/'datos_semi_markov/runs_train.npz');trip=torch.tensor(data['trip_train_index'].astype(np.int64),device='cuda');offset=torch.tensor(data['run_offsets'],device='cuda');cells=torch.tensor(data['cell'].astype(np.int32),device='cuda');duration=torch.tensor(data['duration_n'].astype(np.int64),device='cuda');valid=torch.tensor(np.unpackbits(data['outgoing_valid_bits'])[:len(cells)].astype(bool),device='cuda');assert torch.all(cells[offset[1:]-1].long()==dest[trip]);assert torch.all(duration[offset[1:]-1]>=0);proxy=np.sort(np.random.default_rng(5432).choice(len(v),512,replace=False));np.save(run/'selection_indices.npy',proxy);old=np.load(BEST/'validacion.npz');write(run/'baseline_selection.json',dict(n=512,retained_model_mae_s=metric(v[proxy],old['pred'][proxy])['mae_s'],scope='Selection subset only; final all127941'))
 def local_loss(ti):
  ix=trip[ti];length=offset[ti+1]-offset[ti];ri=offset[ti]+(torch.rand(len(ti),device='cuda')*length).long();terminal=ri==offset[ti+1]-1;cur=cells[ri].long();safe=torch.where(cur<model.N,cur,origin[ix]);nxt=cells[(ri+1).clamp_max(len(cells)-1)].long();q,oo,dd,p0,e,we=model.encode_od(x[ix],origin[ix],dest[ix]);p,wm=model.state_parameters(q,x[ix,:4],safe,dd);dx=torch.div(nxt,grid['ny'],rounding_mode='floor')-torch.div(cur,grid['ny'],rounding_mode='floor');dy=nxt%grid['ny']-cur%grid['ny'];code=(dx+3)*7+(dy+3);act=(code-(code>24)).clamp(0,47);good=valid[ri]&(~terminal)&(cur<model.N)&(nxt<model.N);good&=model.allowed[safe,act];move_d=duration[ri].clamp(1,119)-1;end_d=duration[ri].clamp(0,119)
  route=-(p.gather(1,act[:,None])[:,0].clamp_min(1e-9).log()*good).sum()/good.sum().clamp_min(1);move=-(wm.gather(1,move_d[:,None])[:,0].clamp_min(1e-9).log()*good).sum()/good.sum().clamp_min(1);terminal_mask=terminal&(cur<model.N);end_wait=-(we.gather(1,end_d[:,None])[:,0].clamp_min(1e-9).log()*terminal_mask).sum()/terminal_mask.sum().clamp_min(1);at_dest=(cur==dd)&(cur<model.N);end_ce=(nn.functional.binary_cross_entropy(e.clamp(1e-6,1-1e-6),terminal.float(),reduction='none')*at_dest).sum()/at_dest.sum().clamp_min(1);label=((target[ix]-60)/15).long();nll=-p0.gather(1,label[:,None]).clamp_min(1e-9).log().mean();rps=(p0.cumsum(-1)-(model.duration_grid[None,:]>=target[ix,None]).float()).square().mean();return route+move+end_wait+end_ce+nll+5*rps
 @torch.no_grad()
 def predict(indices=None):
  model.eval();torch.set_float32_matmul_precision('highest');use=np.arange(len(v)) if indices is None else np.asarray(indices);out=[];od=[];route=[];masses=[]
  for first in range(0,len(use),32):
   ix=torch.tensor(use[first:first+32],device='cuda');n=len(ix)
   if n<32:ix=torch.cat([ix,ix[-1:].expand(32-n)])
   p,mass,p0,path=model.posterior(vx[ix],vo[ix],vd[ix]);out.append(model.median(p)[:n].cpu().numpy());od.append(model.median(p0)[:n].cpu().numpy());graph_pred=torch.where(mass>1e-6,model.median(path),model.median(p0));route.append(graph_pred[:n].cpu().numpy());masses.append(mass[:n].cpu().numpy())
  torch.set_float32_matmul_precision('high');return tuple(np.concatenate(z) for z in [out,od,route,masses])
 # Check the actual local data path before the long fit.
 model.train();ix=torch.arange(64,device='cuda');test=local_loss(ix);assert torch.isfinite(test);test.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);model.zero_grad(set_to_none=True);write(run/'verificacion_inicial.json',dict(passed=True,terminal_state_matches_given_destination=True,finite_local_gradients=True,query_route_is_not_input=True,Test_access=False))
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0003);best=float('inf');best_stage=None;history=[];bad=0
 def save(stage,score):
  torch.save(dict(state=model.state_dict(),stats=model_stats,power=power,grid=grid,features=names,config=cfg),run/'modelo.pt');write(run/'mejor.json',dict(stage=stage,selection_validation=score,seconds=time.time()-start))
 for epoch in range(1,a.epochs+1):
  model.train();total=0
  for ti in torch.randperm(len(trip),device='cuda').split(2048):
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):loss=local_loss(ti)
   assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=float(loss.detach())*len(ti)
  r=dict(stage='local',epoch=epoch,train_loss=total/len(trip),seconds=time.time()-start)
  if epoch%2==0 or epoch==1:
   pred,pod,pg,mass=predict(proxy);score=metric(v[proxy],pred);r.update(selection_validation=score,od_selection=metric(v[proxy],pod),graph_selection=metric(v[proxy],pg),mean_absorption_mass=float(mass.mean()))
   if score['mae_s']<best-.1:best=score['mae_s'];bad=0;best_stage=dict(local_epoch=epoch);save(best_stage,score)
   else:bad+=1
  history.append(r);write(run/'historia.json',history);print(json.dumps(r),flush=True)
  if bad>=8 or time.time()-start>cfg['max_seconds']:break
 ck=torch.load(run/'modelo.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['state']);del ck;opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0003)
 for step in range(1,a.fine_steps+1):
  model.train();ti=torch.randint(len(trip),(4,),device='cuda');ix=trip[ti];opt.zero_grad(set_to_none=True);p,mass,p0,path=model.posterior(x[ix],origin[ix],dest[ix]);label=((target[ix]-60)/15).long();loss=-p.gather(1,label[:,None]).clamp_min(1e-9).log().mean()+5*(p.cumsum(-1)-(model.duration_grid[None,:]>=target[ix,None]).float()).square().mean();assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
  if step%100==0 or step==1:
   pred,pod,pg,mass=predict(proxy);score=metric(v[proxy],pred);r=dict(stage='joint_total',step=step,loss=float(loss.detach()),selection_validation=score,od_selection=metric(v[proxy],pod),graph_selection=metric(v[proxy],pg),mean_absorption_mass=float(mass.mean()),seconds=time.time()-start);history.append(r);write(run/'historia.json',history);print(json.dumps(r),flush=True)
   if score['mae_s']<best-.1:best=score['mae_s'];best_stage=dict(fine_step=step);save(best_stage,score)
  if time.time()-start>cfg['max_seconds']:break
 ck=torch.load(run/'modelo.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['state']);pred,pod,pg,mass=predict();np.savez_compressed(run/'validacion.npz',row=va['row'],timestamp=va['timestamp'],true=v,pred=pred,od_head=pod,graph_with_support_fallback=pg,absorption_mass=mass);result=dict(id=a.id,config=cfg,validation=metric(v,pred),od_validation=metric(v,pod),graph_validation=metric(v,pg),best_stage=best_stage,model_sha256=sha(run/'modelo.pt'),data_grid_sha256=sha(OUT/'datos_semi_markov/resultado.json'),code_sha256=sha(__file__),model_code_sha256=sha(OUT/'modelo_espacial.py'),kernel_code_sha256=sha(OUT/'nucleo_semi_markov.py'),paired_vs_retained=paired(v,pred,old['pred'],va['timestamp']),parameters=sum(p.numel() for p in model.parameters()),actual_data_files_accessed=sorted(access),seconds=time.time()-start,query_route_is_input=False,Test_access=False,goal_validation=metric(v,pred)['mae_s']<73.5)
 write(run/'resultado.json',result);print(json.dumps({k:result[k] for k in ['id','validation','od_validation','graph_validation','seconds']}),flush=True)
if __name__=='__main__':main()
