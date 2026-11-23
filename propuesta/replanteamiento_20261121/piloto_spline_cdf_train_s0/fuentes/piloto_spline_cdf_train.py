"""Chronological Train-only flexible-CDF alternative; exact original labels and fivefields."""
from base_replanteo import *
import torch,shutil,gc
from inferencia import feature_input
from modelo_spline_cdf import SplineDurationCDF
from modelo_aux_subviajes import duration_score
access=deny_test();start=time.time();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction(.3)
torch.manual_seed(20261122);torch.cuda.manual_seed_all(20261122)
proof=read(OUT/'verificacion_spline_cdf.json');assert proof['passed'] and proof['model_sha256']==sha(OUT/'modelo_spline_cdf.py')
run=OUT/'piloto_spline_cdf_train_s0';assert not run.exists();run.mkdir();(run/'fuentes').mkdir()
for name in ['piloto_spline_cdf_train.py','modelo_spline_cdf.py','modelo_aux_subviajes.py','base_replanteo.py']:shutil.copy2(OUT/name,run/'fuentes'/name)
plan=dict(scope='ChronologicalheldTrain ODgate, not primaryValidation orconfirmation.',
 hypothesis='Conditionaldurationdensity may need moreflexibility than restricted finiteGaussiantimecomponents. Replace with monotone rationalquadraticsplineCDF and nonlinearperfeature embeddings; no route/history/driveraux.',
 inspiration=dict(title='TabResFlow: A Normalizing Spline Flow Model for Probabilistic Univariate Tabular Regression',venue='ECAI2025, officialpagefirstonline2026-08-25',url='https://journals.sagepub.com/doi/10.3233/FAIA251412',
 adaptation='Inspiredarchitecture, notexactreplica:8ensembles, SiLU/LayerNorm/rawfeatureskip, exactdiscretebinNLL+RPS instead continuousNLL. No publishedTTEperformanceclaim.'),
 architecture=dict(members=8,width=192,residual_blocks=3,numeric_feature_encoder='scalar->8->3 perfeature plusrawskip',bins=32,spline_bound=4,minimum_width_height_derivative=.001,dropout=.1),
 seed=20261122,fit_n=200000,held_n=50000,held_last_days=21,features=206,
 training=dict(epochs=50,batch=1024,lr_initial=.001,lr_final=.0001,weight_decay=.0003,clip=5,loss='OriginaldurationbinNLL+5RPS'),
 precision='NeuralFP32highest; splineinverse/CDF integrationFP64 forstableintervalprobabilities. NoAMP.',
 duration_support='Original60..1785step15; all primarylabels unchanged, no dequantization/relabeling.',
 stopping='Fixed50epochs; evaluateheldonlyafterfinalepoch, noheldmodelselection.',
 control='Exactstored206publicCatBoost800tree200kfit/50kheldcontrol; samequeryrows, labels andpurgedcutoff.',
 gate='At least1secondheldTrainMAEreduction andupperdaybootstrapdeltaCI<0 before any fullprimarytrial.',
 resources='OneGPUfraction.3; nofullCDFdiskbank, <=100MiBnewartifacts.',Validation=False,Test=False)
write(run/'plan.json',plan)
d,X,names=feature_input('train',read(BEST/'resultado.json'));keep=np.asarray([not n.startswith('mm_') for n in names]);assert keep.sum()==206
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index'];y=np.load(DATA/'train_y.npy');ts=d['timestamp'].astype(np.int64)
split=int(ts[eligible].max())-21*86400;fp=eligible[ts[eligible]+y[eligible]<split];hp=eligible[ts[eligible]>=split];rng=np.random.default_rng(20261122)
fit=np.sort(rng.choice(fp,200000,replace=False));held=np.sort(rng.choice(hp,50000,replace=False))
prior=read(OUT/'piloto_reloj_fino_train_s0/resultado.json');ref=dict(np.load(OUT/'piloto_reloj_fino_train_s0/predicciones_held_Train.npz'))
assert hashlib.sha256(np.ascontiguousarray(d['row'][fit]).tobytes()).hexdigest()==prior['selected_fit_rows_sha256']
for key in ['row','timestamp']:assert np.array_equal(ref[key],d[key][held])
assert np.array_equal(ref['true'],y[held])
public=X[:,keep];mean=public[fit].mean(0,dtype=np.float64).astype(np.float32);std=public[fit].std(0,dtype=np.float64).clip(.01).astype(np.float32)
lt=np.log(y[fit].astype(np.float64));lm=float(lt.mean());ls=float(lt.std());np.savez_compressed(run/'normalizacion_Train.npz',mean=mean,std=std,log_mean=lm,log_std=ls)
x=torch.tensor(public[fit],device='cuda');vx=torch.tensor(public[held],device='cuda');target=torch.tensor(y[fit],device='cuda');del X,public
model=SplineDurationCDF(mean,std,lm,ls).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0003)
model.train();p=model(x[:32]);pre=duration_score(p,target[:32],model.duration_grid);assert torch.isfinite(pre);pre.backward();assert all(torch.isfinite(q.grad).all() for q in model.parameters() if q.grad is not None);model.zero_grad(set_to_none=True)
write(run/'verificacion_GPU_inicial.json',dict(passed=True,finite_trueTrain_objective_and_gradients=True,Train_loss=float(pre.detach()),features=206,parameters=sum(q.numel() for q in model.parameters()),Test=False))
history=[]
for epoch in range(1,51):
 model.train();lr=.0001+.0009*.5*(1+np.cos(np.pi*(epoch-1)/49))
 for group in opt.param_groups:group['lr']=float(lr)
 torch.manual_seed(9000+epoch);torch.cuda.manual_seed_all(9000+epoch);order=torch.randperm(len(x),device='cuda',generator=torch.Generator(device='cuda').manual_seed(9000+epoch))
 total=0.;count=0
 for ix in order.split(1024):
  opt.zero_grad(set_to_none=True);lp=model(x[ix]);loss=duration_score(lp,target[ix],model.duration_grid);assert torch.isfinite(loss)
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step();total+=float(loss.detach())*len(ix);count+=len(ix)
 row=dict(epoch=epoch,n=count,loss=total/count,lr=float(lr),seconds=time.time()-start);history.append(row)
 write(run/'historia_Train.json',history)
 if epoch%5==0:print(json.dumps(dict(stage='SplineCDF_Train_only',**row)),flush=True)
model.eval();parts=[]
with torch.no_grad():
 for first in range(0,len(vx),2048):
  xx=vx[first:first+2048];n=len(xx)
  if n<2048:xx=torch.cat([xx,xx[-1:].expand(2048-n,-1)])
  lp=model(xx);parts.append(model.quantiles(lp)[:n].cpu().numpy());del lp
quantiles=np.concatenate(parts);pred=quantiles[:,1];comparison=paired(y[held],pred,ref['control'],ts[held]);comparison['interpretation']='OnefixedfinalTrainonlyCDFtrial, chronologicalheldTrain; noValidationselection orconfirmation.'
passed=bool(comparison['delta_mae_s']<=-1 and comparison['ci95_s'][1]<0)
if passed:torch.save(dict(state=model.state_dict(),plan=plan,features=[n for n,k in zip(names,keep) if k]),run/'modelo.pt')
np.savez_compressed(run/'predicciones_held_Train.npz',row=d['row'][held],timestamp=ts[held],true=y[held],pred=pred,quantiles=quantiles,control=ref['control'])
result=dict(passed=passed,metrics=metric(y[held],pred),control=metric(y[held],ref['control']),comparison=comparison,scope=plan['scope'],
 epochs=50,held_evaluation_only_final=True,fit_arrival_purge=True,unchanged_duration_support_and_labels=True,no_extra_query_information=True,
 sources={p.name:sha(p) for p in (run/'fuentes').glob('*.py')},predictions_sha256=sha(run/'predicciones_held_Train.npz'),
 normalization_sha256=sha(run/'normalizacion_Train.npz'),model_sha256=sha(run/'modelo.pt') if passed else None,Validation_access=False,Test_access=False,seconds=time.time()-start)
write(run/'resultado.json',result);print(json.dumps(dict(passed=passed,metrics=result['metrics'],control=result['control'],delta_mae_s=comparison['delta_mae_s'],seconds=result['seconds'])),flush=True)
