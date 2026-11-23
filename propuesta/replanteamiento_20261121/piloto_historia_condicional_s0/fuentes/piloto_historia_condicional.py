"""CPU-only matched event heads over a frozen native-prefix representation. Train only."""
from base_replanteo import *
from modelo_prefijo_nativo import NativePrefixArrival
from datos_prefijo_nativo import NativeTrainSequences
from historia_eventos import TemporalEventHistory,event_groups
from inferencia import feature_input
import torch,shutil
torch.set_num_threads(2);torch.manual_seed(2026)
access=deny_test()
parent=OUT/'prefijo_nativo_s0'
assert (parent/'resultado.json').exists(),'Wait for frozen full native result'
run=OUT/'piloto_historia_condicional_s0'
assert not run.exists()
run.mkdir()
start=time.time()
parent_sha=sha(parent/'modelo.pt')
history_counts_sha=sha(OUT/'prior_eventos/conteos_fold.npz')
assert history_counts_sha==read(OUT/'prior_eventos/resultado.json')['outputs']['conteos_fold.npz']['sha256']
assert parent_sha==read(parent/'resultado.json')['model_sha256']
ck=torch.load(parent/'modelo.pt',map_location='cpu',weights_only=False)
grid=ck['grid']
graph=dict(np.load(OUT/'datos_semi_markov/grafo_grid.npz'))
distance=np.load(OUT/'datos_semi_markov/distancias_grid.npy')
counts=np.load(OUT/'prior_transiciones/conteos_fold.npz')['counts']
model=NativePrefixArrival(ck['stats'],grid,graph,distance,ck['power'],counts,ck['phase_initialization']).eval()
model.load_state_dict(ck['state'])
for name in ['counts_fold','counts_base_fold','counts_global_fold','counts_total','counts_base','counts_global']:setattr(model,name,torch.empty(0))
del ck,counts
history=TemporalEventHistory(np.load(OUT/'prior_eventos/conteos_fold.npz')['counts'])
tr,X,names=feature_input('train',read(BEST/'resultado.json'))
y=np.load(DATA/'train_y.npy')
def ids(tag):
    xx=(tr[tag+'_lon']+8.62)*111195*np.cos(np.deg2rad(41.16))
    yy=(tr[tag+'_lat']-41.16)*111195
    return np.floor((xx-grid['x0'])/200).astype(np.int64)*grid['ny']+np.floor((yy-grid['y0'])/200).astype(np.int64)
origin,destination=ids('o'),ids('d')
seq=NativeTrainSequences(OUT/'datos_semi_markov/runs_train.npz',graph['neighbors'],graph['allowed'],grid['ny'],model.N)
selected=np.sort(np.random.default_rng(2026).choice(len(seq.trip),25000,replace=False))
hh=[];pp=[];yy=[];legal=[];days=[];fold=[];original=[]
with torch.no_grad():
    for first in range(0,len(selected),128):
        ii,nb=seq.make(selected[first:first+128],origin,destination,y)
        xb=torch.tensor(X[ii],dtype=torch.float32)
        oo=torch.tensor(origin[ii]);dd=torch.tensor(destination[ii])
        cell=torch.tensor(nb['cell']);heading=torch.tensor(nb['heading']);age=torch.tensor(nb['age'])
        B,L=cell.shape
        q,*_=model.encode_od(xb,oo,dd)
        def repeat(v):return v[:,None].expand(B,L,*v.shape[1:]).reshape(B*L,*v.shape[1:])
        local=model.state_representation(repeat(q),repeat(xb[:,:4]),cell.reshape(-1),repeat(dd),heading.reshape(-1)).reshape(B,L,64)
        elapsed=torch.arange(L,dtype=local.dtype)[None].expand(B,-1)/120
        inputs=torch.cat([local,age[:,:,None]/120,elapsed[:,:,None]],-1)
        hist,_=model.prefix_rnn(inputs,q[None])
        state=local+model.history_projection(hist)
        observed=nb['event_mask']&(nb['terminal']|nb['motion_mask'])
        label=np.where(nb['terminal'],2,np.where(nb['action']>0,1,0))
        at_destination=nb['cell']==destination[ii,None]
        allowed_move=graph['allowed'][nb['cell']].any(-1)
        support=np.stack([np.ones_like(at_destination),allowed_move,at_destination],-1)
        group=event_groups(nb['cell'],nb['heading'],destination[ii,None],tr['timestamp'][ii,None],np.arange(L)[None,:],nb['age'])
        f=(tr['timestamp'][ii]//(7*86400))%5
        prior=history.lookup(group,nb['age'],f[:,None])
        stay=model.stay_head(state).sigmoid()[:,:,0]
        stay=torch.where(torch.tensor(allowed_move),stay,torch.ones_like(stay))
        stop=model.stop_head(state).sigmoid()[:,:,0]*torch.tensor(at_destination)
        joint=torch.stack([(1-stop)*stay,(1-stop)*(1-stay),stop],-1)
        hh.append(state[observed].contiguous())
        pp.append(torch.tensor(prior[observed]))
        yy.append(torch.tensor(label[observed]))
        legal.append(torch.tensor(support[observed]))
        original.append(joint[observed].contiguous())
        days.append(np.broadcast_to((tr['timestamp'][ii]//86400)[:,None],(B,L))[observed].copy())
        fold.append(np.broadcast_to(f[:,None],(B,L))[observed].copy())
        if first%4096==0:print(json.dumps(dict(stage='Train_prefix_hidden_in_RAM',completed_trips=first,total_trips=len(selected))),flush=True)
h=torch.cat(hh);prior=torch.cat(pp);target=torch.cat(yy).long();mask=torch.cat(legal);native=torch.cat(original)
day=np.concatenate(days);fold=np.concatenate(fold)
del hh,pp,yy,legal,original,days,model,X,history,seq
fit=torch.tensor(np.flatnonzero(fold!=0));held=torch.tensor(np.flatnonzero(fold==0))
mean=h[fit].mean(0);std=h[fit].std(0).clamp_min(1e-3)
z=(h-mean)/std
del h
cfg=dict(parent_sha256=parent_sha,history_counts_sha256=history_counts_sha,history_code_sha256=sha(OUT/'historia_eventos.py'),Train_trips=25000,valid_point_states=len(z),fit_points=len(fit),held_points=len(held),held_rule='Train departure UTCweek modulo5==0 for newly fitted heads',counts='Each point prior excludes its own complete departure-week fold at every backoff',representation_scope='Parent representation already saw all eligible Train; this is heldout ONLY for new192weight heads, not whole-model generalization',controls=['original frozen native event output','matched new linear3-event head without temporal prior','same linear head plus log temporal prior'],fit='Fixed60 L-BFGS iterations with strong-Wolfe search; L2 .001 on weights, never select by held labels',GPU=False,Test_access=False)
write(run/'protocolo.json',cfg)
results={};held_loss={};head_states={}
with torch.no_grad():
    old=-native[held].gather(-1,target[held,None]).clamp_min(1e-30).log()[:,0]
    held_loss['native']=old.numpy()
    results['native']=dict(held_NLL=float(old.mean()))
for kind in ['control','history']:
    head=torch.nn.Linear(64,3)
    torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)
    offset=torch.zeros_like(prior) if kind=='control' else prior.clamp_min(1e-30).log()
    optimizer=torch.optim.LBFGS(head.parameters(),lr=1,max_iter=60,max_eval=100,line_search_fn='strong_wolfe')
    calls=[0]
    def closure():
        optimizer.zero_grad()
        logits=(head(z[fit])+offset[fit]).masked_fill(~mask[fit],-1e9)
        loss=torch.nn.functional.cross_entropy(logits,target[fit])+.0005*head.weight.square().sum()
        assert torch.isfinite(loss)
        loss.backward()
        calls[0]+=1
        return loss
    optimizer.step(closure)
    with torch.no_grad():
        logits=(head(z[held])+offset[held]).masked_fill(~mask[held],-1e9)
        hl=torch.nn.functional.cross_entropy(logits,target[held],reduction='none')
        held_loss[kind]=hl.numpy()
        results[kind]=dict(held_NLL=float(hl.mean()),optimizer_evaluations=calls[0],weight_norm=float(head.weight.norm()))
        weight=head.weight/std[None]
        bias=head.bias-(weight*mean[None]).sum(-1)
        head_states[kind]=dict(weight=weight.clone(),bias=bias.clone())
    write(run/'resultados_parciales.json',results)
    print(json.dumps(dict(control=kind,**results[kind])),flush=True)
def compare(a,b):
    diff=held_loss[a]-held_loss[b]
    d,di=np.unique(day[held.numpy()],return_inverse=True)
    sums=np.bincount(di,weights=diff);n=np.bincount(di)
    boot=np.random.default_rng(1729).integers(len(n),size=(3000,len(n)))
    ci=np.quantile(sums[boot].sum(1)/n[boot].sum(1),[.025,.975])
    return dict(delta_NLL=float(diff.mean()),CI95_by_Train_day=ci,days=len(d))
versus_control=compare('history','control')
versus_native=compare('history','native')
worth=all(x['delta_NLL']<-.001 and x['CI95_by_Train_day'][1]<0 for x in [versus_control,versus_native])
torch.save(dict(parent_sha256=parent_sha,history_head=head_states['history'],control_head=head_states['control'],feature_mean=mean,feature_std=std,worth_GPU_test=worth),run/'cabezas.pt')
assert sha(parent/'modelo.pt')==parent_sha
assert not torch.cuda.is_initialized()
(run/'fuentes').mkdir()
for name in ['piloto_historia_condicional.py','historia_eventos.py','datos_prefijo_nativo.py']:shutil.copy2(OUT/name,run/'fuentes'/name)
result=dict(protocol=cfg,results=results,history_minus_matched_control=versus_control,history_minus_native=versus_native,worth_GPU_test=worth,interpretation='Conditional event-head likelihood only, not OD time MAE or independent validation of the whole representation. A gain of coarse historical counts alone does not establish complementarity to a neural model.',hidden_bank_saved=False,no_GPU=True,actual_files_accessed=sorted(access),Test_access=False,head_sha256=sha(run/'cabezas.pt'),code_sha256=sha(__file__),seconds=time.time()-start)
write(run/'resultado.json',result)
print(json.dumps({k:result[k] for k in ['results','history_minus_matched_control','history_minus_native','worth_GPU_test','seconds']},default=lambda x:x.tolist()),flush=True)
