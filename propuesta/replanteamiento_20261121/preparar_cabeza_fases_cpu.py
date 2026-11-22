"""CPU-only initialization of compact residence head; no Validation or Test labels."""
from base_replanteo import *
from modelo_dir_fases import PhaseDirectionalArrival
from modelo_direccional import heading_ids
from inferencia import feature_input
import torch,copy
torch.set_num_threads(4);torch.manual_seed(0);access=deny_test();run=OUT/'cabeza_fases_cpu';run.mkdir(exist_ok=False);start=time.time();parent=OUT/'espacial_prior_cf_s0';parent_sha=sha(parent/'modelo.pt');ck=torch.load(parent/'modelo.pt',map_location='cpu',weights_only=False);grid=ck['grid'];graph=dict(np.load(OUT/'datos_semi_markov/grafo_grid.npz'));dist=np.load(OUT/'datos_semi_markov/distancias_grid.npy');counts=np.load(OUT/'prior_transiciones/conteos_fold.npz')['counts'];init=read(OUT/'inicializacion_fases.json');model=PhaseDirectionalArrival(ck['stats'],grid,graph,dist,ck['power'],counts,init).eval();loaded=model.load_shared_state(ck['state']);del counts,ck
tr,X,names=feature_input('train',read(BEST/'resultado.json'));z=np.load(OUT/'datos_semi_markov/runs_train.npz');cell=z['cell'].astype(np.int32);off=z['run_offsets'];trip=z['trip_train_index'];valid=np.unpackbits(z['outgoing_valid_bits'])[:len(cell)].astype(bool);ri=np.random.default_rng(2026).choice(len(cell)-1,400000,replace=False);ti=np.searchsorted(off,ri,side='right')-1;ix=trip[ti];ok=valid[ri]&(ri<off[ti+1]-1)&(cell[ri]<model.N)&(cell[ri+1]<model.N);ri=ri[ok][:200000];ti=ti[ok][:200000];ix=ix[ok][:200000];assert len(ri)==200000
def ids(tag):
 xx=(tr[tag+'_lon']+8.62)*111195*np.cos(np.deg2rad(41.16));yy=(tr[tag+'_lat']-41.16)*111195;return np.floor((xx-grid['x0'])/200).astype(np.int64)*grid['ny']+np.floor((yy-grid['y0'])/200).astype(np.int64)
origin=ids('o');dest=ids('d');fold=((tr['timestamp'][ix]//(7*86400))%5).astype(np.int64);hcache=[]
hook=model.state_encoder.register_forward_hook(lambda module,inputs,output:hcache.append(output.detach().clone()))
with torch.no_grad():
 for first in range(0,len(ri),512):
  rr=ri[first:first+512];ii=ix[first:first+512];tt=ti[first:first+512];cur=torch.tensor(cell[rr],dtype=torch.long);prev=torch.tensor(cell[np.maximum(rr-1,0)],dtype=torch.long);known=torch.tensor((rr>off[tt])&(cell[np.maximum(rr-1,0)]<model.N)&valid[np.maximum(rr-1,0)]);heading=torch.where(known,heading_ids(cur//grid['ny']-prev//grid['ny'],cur%grid['ny']-prev%grid['ny']),4);xx=torch.tensor(X[ii]);q,oo,dd,p0,e,we=model.encode_od(xx,torch.tensor(origin[ii]),torch.tensor(dest[ii]));model.phase_parameters(q,xx[:,:4],cur,dd,heading,torch.tensor(fold[first:first+512]))
  if first%16384==0:print(json.dumps(dict(stage='CPU_frozen_state_encoding',completed_rows=first,total_rows=len(ri),seconds=time.time()-start)),flush=True)
hook.remove();features=torch.cat(hcache);head=copy.deepcopy(model.wait_head);del hcache,model,X,cell,dist,graph
target=torch.tensor(z['duration_n'][ri].astype(np.int64));lags=torch.tensor(init['lags']);exponent=target[:,None]-lags[None,:]-1;mask=exponent>=0;exponent=exponent.clamp_min(0).float();fit=torch.tensor(np.flatnonzero(fold!=0));held=torch.tensor(np.flatnonzero(fold==0));opt=torch.optim.AdamW(head.parameters(),lr=.003,weight_decay=.0001);schedule=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=3,min_lr=1e-5);history=[];best=float('inf');bad=0
def loss(indices):
 raw=head(features[indices]);lw=raw[:,:8].log_softmax(-1);h=raw[:,8:].sigmoid().clamp(.001,.999);lp=lw+h.log()+exponent[indices]*torch.log1p(-h);return -torch.logsumexp(lp.masked_fill(~mask[indices],-torch.inf),-1).mean()
for epoch in range(1,41):
 head.train();total=0.
 for ii in fit[torch.randperm(len(fit))].split(4096):
  opt.zero_grad(set_to_none=True);value=loss(ii);assert torch.isfinite(value);value.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5);opt.step();total+=float(value.detach())*len(ii)
 head.eval()
 with torch.no_grad():score=sum(float(loss(ii))*len(ii) for ii in held.split(4096))/len(held)
 schedule.step(score);r=dict(epoch=epoch,fit_NLL_nats=total/len(fit),held_Train_head_NLL_nats=score,seconds=time.time()-start);history.append(r);write(run/'historia.json',history);print(json.dumps(r),flush=True)
 if score<best-1e-5:
  best=score;bad=0;torch.save(dict(head_state=head.state_dict(),parent_sha256=parent_sha,phase_initialization=init,features=names,seed=2026,sample_run_sha256=hashlib.sha256(ri.tobytes()).hexdigest(),sample_n=len(ri),fit_n=len(fit),held_n=len(held),best_epoch=epoch),run/'cabeza.pt')
 else:bad+=1
 if bad>=8:break
assert sha(parent/'modelo.pt')==parent_sha
write(run/'resultado.json',dict(kind='CPU-only residence-head initialization, NOT travel-time MAE',parent_sha256=parent_sha,head_sha256=sha(run/'cabeza.pt'),best_held_Train_NLL_nats=best,sample_n=len(ri),fit_n=len(fit),held_n=len(held),caveat='Only the new head excludes Train weekfold0; its frozen parent representation was trained with all eligible Train, so this is not an independent whole-model generalization test.',no_disk_hidden_state_bank=True,no_GPU_used=True,Validation_or_Test_labels_used=False,Test_access=False,seconds=time.time()-start,code_sha256=sha(__file__)))
print('CPU compact head initialization finished; no GPU or disk representation bank used.',flush=True)

