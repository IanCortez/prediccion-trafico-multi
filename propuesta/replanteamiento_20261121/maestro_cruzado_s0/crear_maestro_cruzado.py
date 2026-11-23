"""Five-fold privileged probabilistic teachers: each Train target is genuinely out of fit."""
from base_replanteo import *
from features import features
from catboost import CatBoostRegressor
from scipy.special import ndtr
import shutil,argparse
p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');args=p.parse_args()
access=deny_test();assert read(OUT/'piloto_maestro_probabilistico_s0/resultado.json')['worth_building_full_crossfit_teacher']
start=time.time();run=OUT/'maestro_cruzado_s0'
plan=dict(scope='Privileged soft-target creation ONLY for Train, never primary query prediction',
 sample='All1023485 embargoe-safe Train rows, no quality/target-based row removal',
 folds='UTC departure week modulo5. Every row is predicted exactly once by a teacher fit on other4 folds.',
 teacher='CatBoost RMSEWithUncertainty on log(T),1000 fixed trees,depth7,LR.07,L2=5,seed20261122+fold,CPU2threads,no held early stopping or iteration selection',
 features='39 deterministic OD/calendar features plus actual path log length,km length,14 normalized-arclength deviations and finite flag. No CALL_TYPE, number of points, max GPS step or fitted shared feature banks.',
 supervision='T remains the original duration label. Observed geometry is privileged Train-only teacher input. Eventual student forward is five-field OD only.',
 storage='Five compact CBM teachers and one compressed NPZ of two raw lognormal parameters per Train row; no per-trip CDF or hidden bank.',
 inference='RawFormulaVal=(mu_logT,log_sigma); sigma fixed clipped[.05,1.5],116 support bins60..1785. Quantization and fitted model statistics use only each fit partition.',
 later_student='Predeclare auxiliary soft expected NLL+5RPS weight0.2 on the original strong model, primary/geometry objectives unchanged,4 fixed epochs,LR1e-5. Reuse the already queued aux_solicitud_control_s0 only after verifying exact architecture/data/seed/budget/primary-loss equivalence; otherwise do not claim matched effect.',
 GPU='None. Existing risk and request-type study keep priority. No student GPU launch without CPU probability/gradient and actual GPU gates.',
 Test='sealed',Validation='not opened')
if not args.resume:
 assert not run.exists();run.mkdir();shutil.copy2(__file__,run/Path(__file__).name);write(run/'plan.json',plan)
else:
 assert read(run/'plan.json')==plan and sha(__file__)==sha(run/Path(__file__).name)
d=dict(np.load(DATA/'train_x.npz'));eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index']
assert len(eligible)==1023485
part={k:v[eligible] for k,v in d.items()};X,names=features(part,enriched=False,calendar=True);assert len(names)==39 and all(not n.startswith('call_') for n in names)
y=np.load(DATA/'train_y.npy')[eligible];fold=(part['timestamp']//(7*86400))%5
geo=dict(np.load(DATA/'geometria_train.npz'));pos=np.searchsorted(geo['row'],part['row'])
assert (pos<len(geo['row'])).all() and np.array_equal(geo['row'][pos],part['row'])
g=geo['X'][pos,:15].copy();finite=np.isfinite(g).all(1);g[~finite]=0
F=np.column_stack([X,np.log1p(np.maximum(g[:,0],0)),g[:,0]/1000,g[:,1:],finite.astype(float)]).astype(np.float32)
assert F.shape==(len(y),56) and np.isfinite(F).all()
del d,X,geo,g
raw=np.full((len(y),2),np.nan,np.float32);assigned=np.zeros(len(y),np.uint8);records=[]
grid=np.arange(60,1786,15,dtype=np.float64);edges=np.log(np.arange(67.5,1785,15,dtype=np.float64))
def summarize(parameters,truth):
 sigma=np.exp(parameters[:,1].astype(np.float64)).clip(.05,1.5);mu=parameters[:,0].astype(np.float64)
 cdf=ndtr((edges[None,:]-mu[:,None])/sigma[:,None]);prob=np.diff(np.column_stack([np.zeros(len(truth)),cdf,np.ones(len(truth))]),axis=1);prob=np.maximum(prob,1e-9);prob/=prob.sum(1,keepdims=True);cdf=prob.cumsum(1)
 label=((truth-60)/15).astype(int);q=np.column_stack([grid[(cdf<level).sum(1).clip(0,115)] for level in [.1,.5,.9]])
 return dict(duration=metric(truth,q[:,1]),NLL=float(-np.log(prob[np.arange(len(truth)),label]).mean()),CRPS_s=float(15*np.square(cdf-(grid[None,:]>=truth[:,None])).sum(1).mean()),coverage80=float(np.mean((truth>=q[:,0])&(truth<=q[:,2]))),sigma_mean=float(sigma.mean()))
for held_fold in range(5):
 held=np.flatnonzero(fold==held_fold);fit=np.flatnonzero(fold!=held_fold)
 assert len(np.intersect1d(held,fit))==0 and len(held)+len(fit)==len(y)
 assert np.all(fold[fit]!=held_fold) and np.all(fold[held]==held_fold)
 model_file=run/('maestro_fold_'+str(held_fold)+'.cbm');pred_file=run/('parametros_fold_'+str(held_fold)+'.npz')
 if pred_file.exists():
  saved=dict(np.load(pred_file));assert np.array_equal(saved['row'],part['row'][held]);assert sha(model_file)==read(run/('fold_'+str(held_fold)+'.json'))['model_sha256']
  pr=saved['raw'];assert pr.shape==(len(held),2)
 else:
  model=CatBoostRegressor(iterations=1000,depth=7,learning_rate=.07,l2_leaf_reg=5,loss_function='RMSEWithUncertainty',random_seed=20261122+held_fold,thread_count=2,allow_writing_files=False,verbose=False,task_type='CPU')
  if model_file.exists():model.load_model(str(model_file))
  else:
   model.fit(F[fit],np.log(y[fit].astype(np.float64)));model.save_model(str(model_file))
  pr=model.predict(F[held],prediction_type='RawFormulaVal').astype(np.float32)
  assert np.isfinite(pr).all() and pr.shape==(len(held),2)
  np.savez_compressed(pred_file,row=part['row'][held],raw=pr)
 raw[held]=pr;assigned[held]+=1
 metrics=summarize(pr,y[held]);record=dict(fold=held_fold,fit_n=len(fit),held_n=len(held),own_week_fold_excluded=True,
  fit_rows_sha256=hashlib.sha256(part['row'][fit].tobytes()).hexdigest(),held_rows_sha256=hashlib.sha256(part['row'][held].tobytes()).hexdigest(),
  model_sha256=sha(model_file),held_parameters_sha256=sha(pred_file),held_Train=metrics,elapsed_seconds=time.time()-start)
 records.append(record);write(run/('fold_'+str(held_fold)+'.json'),record)
 write(run/'progreso.json',dict(completed_folds=records,goal='active',Test='sealed',no_GPU=True))
 print(json.dumps(dict(stage='out_of_fit_teacher_fold_complete',**record)),flush=True)
assert np.all(assigned==1) and np.isfinite(raw).all()
target=run/'parametros_train_fuera_bloque.npz'
np.savez_compressed(target,row=part['row'],timestamp=part['timestamp'],trip_train_index=eligible,fold=fold.astype(np.uint8),mu_log_seconds=raw[:,0],log_sigma=raw[:,1])
saved=dict(np.load(target));assert np.array_equal(saved['row'],part['row'])
assert np.array_equal(saved['mu_log_seconds'],raw[:,0]) and np.array_equal(saved['log_sigma'],raw[:,1])
# Score all rows in bounded chunks without storing CDF banks.
aggregate=[]
for first in range(0,len(y),16384):
 last=min(first+16384,len(y));aggregate.append((last-first,summarize(raw[first:last],y[first:last])))
summary={key:sum(n*s[key] for n,s in aggregate)/len(y) for key in ['NLL','CRPS_s','coverage80','sigma_mean']}
result=dict(scope=plan['scope'],Train_n=len(y),all_eligible_rows_assigned_once=True,own_whole_fold_excluded_for_every_row=True,folds=records,full_out_of_fit_Train_distribution_scores=summary,
 primary_OD_MAE_not_measured=True,Validation_access=False,Test_access=False,no_GPU=True,no_CDF_bank_on_disk=True,
 raw_parameter_dtype='float32',parameters_sha256=sha(target),parameter_file_bytes=target.stat().st_size,
 inputs={name:sha(DATA/name) for name in ['train_x.npz','train_y.npy','geometria_train.npz']},
 source_sha256=sha(__file__),actual_files_accessed=sorted(access),seconds=time.time()-start,
 next='Only auxiliary Train targets. Existing request study retains priority; evaluate later OD-only student against exact matched control, with all original labels and cohort unchanged.')
assert all('validation' not in s.lower() and 'test' not in s.lower() for s in access)
write(run/'resultado.json',result)
removed=[]
for held_fold in range(5):
 file=run/('parametros_fold_'+str(held_fold)+'.npz')
 z=dict(np.load(file));held=np.flatnonzero(fold==held_fold);assert np.array_equal(z['row'],saved['row'][held]);assert np.array_equal(z['raw'],raw[held])
 assert file.resolve().is_relative_to(OUT.resolve());removed.append(dict(file=file.name,bytes=file.stat().st_size));file.unlink()
write(run/'intermedios_retirados.json',dict(removed=removed,final_parameter_archive_verified=True,teachers_retained=True))
print(json.dumps(result),flush=True)
