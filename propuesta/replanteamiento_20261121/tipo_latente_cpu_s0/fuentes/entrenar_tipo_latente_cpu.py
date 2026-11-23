"""Frozen strong encoder + supervised latent type duration tilts, matched CPU control."""
from base_replanteo import *
import torch,shutil,copy,gc
from inferencia import feature_input
from modelo_mezcla_rutas import make_duration_model
from modelo_tipo_latente import LatentTypeTilt,duration_median
access=deny_test();torch.set_num_threads(2);torch.manual_seed(20261122)
torch.set_float32_matmul_precision('highest')
run=OUT/'tipo_latente_cpu_s0';assert not run.exists()
assert read(OUT/'piloto_tipo_condicional_s0/resultado.json')['worth_full_experiment']
run.mkdir();(run/'fuentes').mkdir();start=time.time()
plan=dict(architecture='frozen_TabM_mean_hidden + three semantic duration experts',
 equation='p(T|x) = sum_c p(c|h(x)) softmax_T(log p_reference(T|x) + b(T)^T a_c(h(x)))',
 experts='A/B/C, 8 fixed smooth log-duration basis functions per expert; 64 hidden units; bounded coefficients 1.5*tanh(raw/1.5)',
 auxiliary='Train CALL_TYPE target for joint type and conditional duration NLL; never a forward input',
 main_loss='supervised: -log p(c|x)-log p(T|x,c)+5*RPS(marginal); matched control: -log p(T|x)+5*RPS(marginal)',
 backbone='Strong reference entirely frozen, including 216 feature preprocessing and physical anchors; 16 member hidden states averaged.',
 Train='All 1,023,485 embargoe-safe Train rows used for new head fitting; no new query filters',
 Validation='All unchanged 127,941 rows, only after final fixed heads frozen',
 Test='sealed',device='CPU only',threads=2,canonical_batch=32,
 initial='zero tilt and uniform type gate reproduce base distribution; seeds and matched architecture identical',
 budget=dict(steps=12000,batch=256,lr=.0005,AdamW_weight_decay=.001,gradient_clip=1,held_early_stopping=False),
 checkpoint='Final fixed step only. No search of blend weights and no Validation selection during head fitting.',
 uncertainty='Day-block paired bootstrap 3000, exploratory after adaptive repeated Validation use.',
 conservation='Any full MAE below 131.06 seconds and any new best retains heads, metrics, predictions, exact sources, frozen-parent references and hashes.',
 resources='Preallocated Train hidden and base probability arrays in RAM only, about 2.6GB; no CDF/hidden banks on disk; one other GPU job unaffected.',
 caveat='Base sees all Train, so residual-head fitting uses in-sample base representations; Train pilot gate is not independent generalization evidence.')
write(run/'plan.json',plan)
for f in ['entrenar_tipo_latente_cpu.py','modelo_tipo_latente.py','base_replanteo.py']:
 shutil.copy2(OUT/f,run/'fuentes'/f)
write(run/'fuentes/parent_sources.json',{str(p.relative_to(FROZEN)):sha(p) for p in FROZEN.rglob('*.py')})
reference_hash=sha(BEST/'modelo.pt');rec=read(BEST/'resultado.json')
ck=torch.load(BEST/'modelo.pt',map_location='cpu',weights_only=False)
parent=make_duration_model(ck,rec['config']);parent.load_state_dict(ck['state']);parent.eval()
for p in parent.parameters():p.requires_grad_(False)
del ck
latest=[]
def capture(module,args):latest.append(args[0].detach().mean(1))
hook=parent.backbone.output.register_forward_pre_hook(capture)
d,X,names=feature_input('train',rec)
assert names==rec['features'] and all(not n.startswith('call_') for n in names)
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index']
assert len(eligible)==1023485
Y=torch.from_numpy(np.load(DATA/'train_y.npy')[eligible].copy())
C=torch.from_numpy(np.searchsorted(np.array(['A','B','C']),d['call'][eligible]).astype(np.int64))
assert np.isin(d['call'][eligible],['A','B','C']).all()
del d
hidden=torch.empty((len(eligible),512),dtype=torch.float32)
base=torch.empty((len(eligible),116),dtype=torch.float32)
with torch.no_grad():
 for at in range(0,len(eligible),plan['canonical_batch']):
    ids=eligible[at:at+plan['canonical_batch']]
    out=parent(torch.from_numpy(X[ids]))
    assert len(latest)==1
    hidden[at:at+len(ids)]=latest.pop()
    p=out[0].exp().mean(1);base[at:at+len(ids)]=p/p.sum(-1,keepdim=True)
    if at%65536==0:
      print(json.dumps(dict(stage='CPU_Train_representation_RAM_only',rows=min(at+len(ids),len(eligible)),seconds=time.time()-start)),flush=True)
del X,eligible,out,p;gc.collect()
assert torch.isfinite(hidden).all() and torch.isfinite(base).all()
mean=hidden.mean(0);std=hidden.std(0).clamp_min(.1);L=((Y-60)/15).long()
heads={};hist={}
initial=LatentTypeTilt(mean,std)
with torch.no_grad():
 pp,gg,qq=initial(hidden[:64],base[:64])
 initial_error=float((pp.exp()-base[:64]).abs().max())
 assert initial_error<1e-6
 assert torch.equal(duration_median(pp),duration_median(base[:64].log()))
for supervised in [False,True]:
 name='supervised_type' if supervised else 'matched_unsupervised'
 torch.manual_seed(20261122);head=LatentTypeTilt(mean,std)
 opt=torch.optim.AdamW(head.parameters(),lr=plan['budget']['lr'],weight_decay=plan['budget']['AdamW_weight_decay'])
 rg=torch.Generator().manual_seed(17);head.train();history=[]
 for step in range(1,plan['budget']['steps']+1):
    ids=torch.randint(len(Y),(plan['budget']['batch'],),generator=rg)
    lp,g,q=head(hidden[ids],base[ids])
    desired=(head.grid[None,:]>=Y[ids,None]).float()
    rps=(lp.exp().cumsum(-1)-desired).square().mean()
    nll=-lp.gather(1,L[ids,None]).mean()
    if supervised:main=-(g[torch.arange(len(ids)),C[ids]]+q[torch.arange(len(ids)),C[ids],L[ids]]).mean()
    else:main=nll
    loss=main+5*rps;assert torch.isfinite(loss)
    opt.zero_grad(set_to_none=True);loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1,error_if_nonfinite=True);opt.step()
    if step%1000==0:
      item=dict(stage=name,step=step,loss=float(loss.detach()),marginal_NLL=float(nll.detach()),seconds=time.time()-start)
      history.append(item);print(json.dumps(item),flush=True)
    if step%3000==0:
      torch.save(dict(head=head.state_dict(),optimizer=opt.state_dict(),step=step,supervised=supervised,parent_sha256=reference_hash,rng_state=rg.get_state(),torch_rng_state=torch.get_rng_state(),plan=plan),run/(name+'_reanudacion.pt'))
 head.eval();heads[name]=head
 torch.save(dict(state=head.state_dict(),parent_sha256=reference_hash,parent=str(BEST.relative_to(ROOT)),parent_source_sha256=read(run/'fuentes/parent_sources.json'),plan=plan,supervised=supervised),run/(name+'.pt'))
 hist[name]=history
write(run/'historia.json',hist)
del hidden,base,Y,C,L,opt;gc.collect()
va,V,names=feature_input('validation',rec);assert names==rec['features']
y=np.load(DATA/'validation_y.npy');assert len(y)==127941
frozen=dict(np.load(BEST/'validacion.npz'));assert np.array_equal(frozen['row'],va['row']) and np.array_equal(frozen['true'],y)
pred={name:np.empty(len(y),np.float32) for name in heads};cpu_ref=np.empty(len(y),np.float32);nlls={name:0. for name in heads}
with torch.no_grad():
 for at in range(0,len(y),plan['canonical_batch']):
    vblock=torch.from_numpy(V[at:at+plan['canonical_batch']])
    out=parent(vblock);h=latest.pop();assert not latest
    p=out[0].exp().mean(1);p=p/p.sum(-1,keepdim=True)
    cpu_ref[at:at+len(vblock)]=duration_median(p.log()).numpy()
    label=torch.from_numpy(((y[at:at+len(vblock)]-60)/15).astype(np.int64))
    for name,head in heads.items():
       lp,g,q=head(h,p);pred[name][at:at+len(vblock)]=duration_median(lp).numpy()
       nlls[name]+=float(-lp.gather(1,label[:,None]).double().sum())
    if at%32768==0:print(json.dumps(dict(stage='full_Validation_CPU',rows=min(at+len(vblock),len(y)),seconds=time.time()-start)),flush=True)
hook.remove()
# Order/singleton checks on declared CPU configuration, after final heads frozen.
def point_probe(indices,batch):
 values={name:[] for name in heads}
 local=[]
 hh=parent.backbone.output.register_forward_pre_hook(lambda module,args:local.append(args[0].detach().mean(1)))
 with torch.no_grad():
  for first in range(0,len(indices),batch):
   ids=indices[first:first+batch];out=parent(torch.from_numpy(V[ids]));h=local.pop()
   p=out[0].exp().mean(1);p=p/p.sum(-1,keepdim=True)
   for name,head in heads.items():values[name].append(duration_median(head(h,p)[0]).numpy())
 hh.remove()
 return {name:np.concatenate(v) for name,v in values.items()}
ix=np.arange(65)[::-1];rev=point_probe(ix,32);single=point_probe(np.array([0,1,63,64]),1)
order_checks={}
for name in heads:
 order_checks[name]=dict(reverse_65_equal=bool(np.array_equal(rev[name],pred[name][ix])),singletons_equal=bool(np.array_equal(single[name],pred[name][[0,1,63,64]])))
reference_equal=bool(np.array_equal(cpu_ref,frozen['pred']))
write(run/'contratos.json',dict(reference_CPU_equal_protected=reference_equal,reference_changed_rows=int(np.sum(cpu_ref!=frozen['pred'])),reference_CPU_metric=metric(y,cpu_ref),initial_max_probability_error=initial_error,order_checks=order_checks,forward_no_observed_type=True,all_Validation_rows_evaluated=True))
np.savez_compressed(run/'validacion.npz',row=va['row'],timestamp=va['timestamp'],true=y,reference=frozen['pred'],reference_CPU=cpu_ref,**pred)
outputs={};qualified=[]
for name,prediction in pred.items():
 m=metric(y,prediction);raw_ok=reference_equal and all(order_checks[name].values())
 outputs[name]=dict(validation=m,duration_NLL=nlls[name]/len(y),paired_vs_reference=paired(y,prediction,frozen['pred'],va['timestamp']),raw_inference_contracts_passed=raw_ok,model_sha256=sha(run/(name+'.pt')))
 if m['mae_s']<131.06 or m['mae_s']<rec['validation']['mae_s']:qualified.append(dict(name=name,mae_s=m['mae_s'],status='qualified' if raw_ok else 'provisional_requires_numerical_reproduction',preserved=True))
u='matched_unsupervised';s='supervised_type'
result=dict(id=run.name,architecture=plan['architecture'],query_information=['o_lon','o_lat','d_lon','d_lat','timestamp'],Train_n=1023485,Validation_n=len(y),actual_type_in_forward=False,outputs=outputs,paired_semantic_vs_matched=paired(y,pred[s],pred[u],va['timestamp']),qualified_candidates=qualified,reference_CPU_equal_protected=reference_equal,Test_access=False,actual_files_accessed=sorted(access),parent_sha256=reference_hash,parent_unchanged=sha(BEST/'modelo.pt')==reference_hash,sources={p.name:sha(p) for p in (run/'fuentes').glob('*.py')},predictions_sha256=sha(run/'validacion.npz'),elapsed_seconds=time.time()-start,goal_status='active unless eligible confirmation <73.5 is subsequently verified')
assert result['parent_unchanged']
for name in heads:
 resume=run/(name+'_reanudacion.pt')
 if resume.exists():
  assert resume.resolve().is_relative_to(OUT.resolve());resume.unlink()
write(run/'resultado.json',result)
write(OUT/'conclusion_tipo_latente_cpu.json',dict(result_file=str((run/'resultado.json').relative_to(ROOT)),outputs=outputs,qualified_candidates=qualified,conclusion='Read full matched-control results; this is a fixed-encoder semantic auxiliary experiment, not Test confirmation.',goal='active',Test='sealed'))
print(json.dumps(result,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else str(v)),flush=True)
