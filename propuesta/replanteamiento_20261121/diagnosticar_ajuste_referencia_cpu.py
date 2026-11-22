"""Approximate fixed-model fit/shift diagnostic; CPU only, no model fitting or Test."""
from base_replanteo import *
from modelo_mezcla_rutas import make_duration_model
from inferencia import feature_input
import torch
torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');access=deny_test();start=time.time();rec=read(BEST/'resultado.json');ck=torch.load(BEST/'modelo.pt',map_location='cpu',weights_only=False);model=make_duration_model(ck,rec['config']).eval();model.load_state_dict(ck['state']);del ck;tr,X,names=feature_input('train',rec);va,V,vnames=feature_input('validation',rec);assert names==vnames==rec['features'];y=np.load(DATA/'train_y.npy');v=np.load(DATA/'validation_y.npy');old=np.load(BEST/'validacion.npz');rg=np.random.default_rng(2026);recent=np.flatnonzero(tr['timestamp']>=tr['timestamp'].max()-30*86400);groups=[('Train_random',X,y,rg.choice(len(y),8192,replace=False)),('Train_recent30days',X,y,rg.choice(recent,min(len(recent),8192),replace=False)),('Validation',V,v,rg.choice(len(v),8192,replace=False))];results={}
with torch.no_grad():
 for name,features,target,idx in groups:
  pred=[];nll=[]
  for first in range(0,len(idx),128):
   ii=idx[first:first+128];out=model(torch.tensor(features[ii]))[0];p=out.exp().mean(1);label=torch.tensor(((target[ii]-60)/15).astype(np.int64));pred.append(model.quantiles(out,(.5,))[:,0].numpy());nll.append((-p.gather(1,label[:,None])[:,0].clamp_min(1e-9).log()).numpy())
  pp=np.concatenate(pred);rr=dict(metric=metric(target[idx],pp),mean_NLL_nats=float(np.concatenate(nll).mean()),label_mean_s=float(target[idx].mean()),label_median_s=float(np.median(target[idx])))
  if name=='Validation':rr['CPU_vs_saved_GPU_changed_predictions']=int((pp!=old['pred'][idx]).sum());rr['saved_GPU_sample']=metric(target[idx],old['pred'][idx])
  results[name]=rr;write(OUT/'diagnostico_ajuste_referencia_cpu_progreso.json',results);print(json.dumps(dict(group=name,**rr)),flush=True)
assert not torch.cuda.is_initialized();report=dict(scope='8192 fixed random examples per group; CPU evaluation of protected frozen model, not new candidate or full-cohort result',results=results,caveat='Train is in sample; distributions differ over time. A small fit gap does not establish a Bayes lower bound; a large gap does not prove a specific cause. CPU arithmetic is checked against saved GPU predictions on the same Validation sample.',code_sha256=sha(__file__),no_GPU_used=True,Test_access=False,seconds=time.time()-start)
write(OUT/'diagnostico_ajuste_referencia_cpu.json',report)

