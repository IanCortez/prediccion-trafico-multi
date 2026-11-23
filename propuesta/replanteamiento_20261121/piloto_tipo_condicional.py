"""Matched CPU-only auxiliary-type test on a frozen strong representation."""
from base_replanteo import *
import torch,shutil
from inferencia import feature_input
from modelo_mezcla_rutas import make_duration_model
from modelo_tipo_latente import LatentTypeTilt,duration_median
access=deny_test();torch.set_num_threads(2);torch.manual_seed(20261122)
torch.set_float32_matmul_precision('highest')
run=OUT/'piloto_tipo_condicional_s0';assert not run.exists();run.mkdir()
start=time.time()
plan=dict(scope='Train-only conditional instrument, not a primary MAE result',
 sample_n=25000,seed=20261122,fit='UTC week modulo 5 != 0',held='UTC week modulo 5 == 0',
 frozen_representation='Strong reference saw all Train. Held rows reserve new heads only, not truly unseen foundation data.',
 comparison=['identical capacity latent type mixture, no class supervision','same mixture, supervised joint duration and CALL_TYPE loss'],
 query=['o_lon','o_lat','d_lon','d_lat','timestamp'],
 actual_type='Train auxiliary target only; forward never accepts it',
 steps=1200,batch=256,lr=.0005,checkpoint='fixed final step, no held stopping/selection',
 GPU=False,Val=False,Test=False,output_CDF_banks_on_disk=False,
 gate='Candidate must beat matched control AND frozen reference in held-Train MAE with upper day-bootstrap delta CI <0, and not worsen held-Train duration NLL versus control. Positive gate only justifies a later independent full-cohort experiment.')
write(run/'plan.json',plan)
refhash=sha(BEST/'modelo.pt')
rec=read(BEST/'resultado.json');d,X,names=feature_input('train',rec)
assert names==rec['features'] and not any(n.startswith('call_') for n in names)
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index']
ii=np.sort(np.random.default_rng(20261122).choice(eligible,plan['sample_n'],replace=False))
features=X[ii].copy();del X
target=np.load(DATA/'train_y.npy')[ii]
labels=np.searchsorted(np.array(['A','B','C']),d['call'][ii])
assert np.isin(d['call'][ii],['A','B','C']).all()
rows=d['row'][ii].copy();stamp=d['timestamp'][ii].copy()
fold=(stamp//(7*86400))%5;fit=fold!=0;held=fold==0
del d
ck=torch.load(BEST/'modelo.pt',map_location='cpu',weights_only=False)
model=make_duration_model(ck,rec['config']);model.load_state_dict(ck['state']);model.eval()
assert all(p.device.type=='cpu' for p in model.parameters())
hs=[];ps=[];latest=[]
def capture(module,args):
    latest.append(args[0].detach().mean(1).clone())
hook=model.backbone.output.register_forward_pre_hook(capture)
with torch.no_grad():
 for at in range(0,len(features),32):
    out=model(torch.from_numpy(features[at:at+32]))
    assert len(latest)==1
    hs.append(latest.pop());ps.append(out[0].exp().mean(1))
    if at%4096==0:print(json.dumps(dict(stage='frozen_Train_CPU',rows=min(at+32,len(features)),seconds=time.time()-start)),flush=True)
hook.remove();hidden=torch.cat(hs);base=torch.cat(ps)
del model,ck,hs,ps,features
assert torch.isfinite(hidden).all() and torch.isfinite(base).all()
base=base/base.sum(-1,keepdim=True)
fit_idx=torch.from_numpy(np.flatnonzero(fit));held_idx=torch.from_numpy(np.flatnonzero(held))
mean=hidden[fit_idx].mean(0);std=hidden[fit_idx].std(0).clamp_min(.1)
Y=torch.tensor(target);C=torch.tensor(labels,dtype=torch.long)
L=((Y-60)/15).long()
h=LatentTypeTilt(mean,std)
with torch.no_grad():
    p,g,q=h(hidden[:64],base[:64]);maxerr=float((p.exp()-base[:64]).abs().max())
    assert maxerr<1e-6
    assert torch.equal(duration_median(p),duration_median(base[:64].log()))
    perm=torch.arange(63,-1,-1)
    pr=h(hidden[:64][perm],base[:64][perm])[0][perm]
    assert torch.max(torch.abs(pr-p))<1e-5
write(run/'contratos.json',dict(zero_residual_max_probability_error=maxerr,initial_medians_unchanged=True,forward_has_no_observed_type=True,permutation_test=True,CPU_only=True))
preds={};nlls={};states={}
base_pred=duration_median(base[held_idx].log()).numpy()
held_y=target[held];held_stamp=stamp[held]
ref_metric=metric(held_y,base_pred)
for supervised in [False,True]:
    name='supervised_type' if supervised else 'matched_unsupervised'
    torch.manual_seed(20261122);head=LatentTypeTilt(mean,std)
    opt=torch.optim.AdamW(head.parameters(),lr=plan['lr'],weight_decay=.001)
    rg=torch.Generator().manual_seed(17)
    head.train()
    for step in range(1,plan['steps']+1):
        index=fit_idx[torch.randint(len(fit_idx),(plan['batch'],),generator=rg)]
        logp,gate,conditional=head(hidden[index],base[index])
        nll=-logp.gather(1,L[index,None]).mean()
        desired=(head.grid[None,:]>=Y[index,None]).float()
        rps=(logp.exp().cumsum(-1)-desired).square().mean()
        if supervised:
            class_logp=gate.gather(1,C[index,None])[:,0]
            cond_logp=conditional[torch.arange(len(index)),C[index],L[index]]
            main=-(class_logp+cond_logp).mean()
        else:main=nll
        loss=main+5*rps
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(),1,error_if_nonfinite=True);opt.step()
        if step%300==0:print(json.dumps(dict(stage=name,step=step,loss=float(loss.detach()),seconds=time.time()-start)),flush=True)
    head.eval();pv=[];lv=[];gv=[]
    with torch.no_grad():
      for block in held_idx.split(256):
        logp,g,q=head(hidden[block],base[block])
        pv.append(duration_median(logp).numpy())
        lv.append(-logp.gather(1,L[block,None])[:,0].numpy())
        gv.append(g.exp().numpy())
    preds[name]=np.concatenate(pv);nlls[name]=np.concatenate(lv);states[name]=head.state_dict()
    print(json.dumps(dict(stage='held_Train_fixed_head',name=name,validation=metric(held_y,preds[name]),duration_NLL=float(nlls[name].mean()),seconds=time.time()-start)),flush=True)
u='matched_unsupervised';s='supervised_type'
delta_control=paired(held_y,preds[s],preds[u],held_stamp)
delta_ref=paired(held_y,preds[s],base_pred,held_stamp)
worth=bool(delta_control['ci95_s'][1]<0 and delta_ref['ci95_s'][1]<0 and nlls[s].mean()<=nlls[u].mean())
np.savez_compressed(run/'held_Train_predicciones.npz',row=rows[held],timestamp=held_stamp,true=held_y,reference=base_pred,matched_unsupervised=preds[u],supervised_type=preds[s])
torch.save(dict(states=states,plan=plan,parent_sha256=refhash),run/'cabezas.pt')
for f in ['piloto_tipo_condicional.py','modelo_tipo_latente.py','base_replanteo.py']:shutil.copy2(OUT/f,run/f)
assert sha(BEST/'modelo.pt')==refhash
assert all('validation' not in p.lower() and 'test' not in p.lower() for p in access)
result=dict(scope=plan['scope'],frozen_representation_limitation=plan['frozen_representation'],
 fit_n=int(fit.sum()),held_n=int(held.sum()),reference_held_Train=ref_metric,
 matched_unsupervised=metric(held_y,preds[u]),supervised_type=metric(held_y,preds[s]),
 paired_vs_control=delta_control,paired_vs_reference=delta_ref,
 duration_NLL={k:float(v.mean()) for k,v in nlls.items()},
 worth_full_experiment=worth,
 interpretation='Matched instrument only. Actual call is a training target, not a primary query feature. It cannot establish new full Validation MAE or confirmation performance.',
 CPU_only=True,Test_access=False,Validation_access=False,actual_files_accessed=sorted(access),
 parent_unchanged=True,parent_sha256=refhash,
 code_sha256=sha(__file__),model_code_sha256=sha(OUT/'modelo_tipo_latente.py'),
 predictions_sha256=sha(run/'held_Train_predicciones.npz'),heads_sha256=sha(run/'cabezas.pt'),
 seconds=time.time()-start)
write(run/'resultado.json',result);print(json.dumps(result,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else str(v)),flush=True)
