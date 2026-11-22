"""Train-only competing stay/move/end hazards with held-week count controls."""
from base_replanteo import *
import torch
torch.set_num_threads(2)
access=deny_test()
start=time.time()
H=120

def point_counts(hist):
    # Each run has one event at age d, with d previous stay decisions.
    events=hist.sum(-1,dtype=np.float64)
    risk=np.cumsum(events[:,::-1],axis=1)[:,::-1]
    return np.stack([risk-events,hist[:,:,0],hist[:,:,1]],-1)

def run_losses(probability,group,age,event):
    lp=np.log(np.maximum(probability,1e-30)).astype(np.float64)
    cum=np.concatenate([np.zeros((len(lp),1)),np.cumsum(lp[:,:,0],axis=1)],1)
    loss=np.empty(len(group),np.float64)
    for first in range(0,len(group),500000):
        s=slice(first,first+500000)
        loss[s]=-cum[group[s],age[s]]-lp[group[s],age[s],event[s]+1]
    return loss

# A run moving at age2 gives two stay targets and one move. A terminal
# run stopping at age0 gives only one end target; event age is not a horizon.
toy=np.zeros((1,4,2),np.uint32)
toy[0,2,0]=1;toy[0,0,1]=1
tc=point_counts(toy)
assert np.array_equal(tc[0],[[1,0,1],[1,0,0],[0,1,0],[0,0,0]])
tp=np.full((1,4,3),1/3.)
tl=run_losses(tp,np.array([0,0]),np.array([2,0]),np.array([0,1]))
assert np.allclose(tl,np.array([3,1])*np.log(3),atol=1e-12)
tr=dict(np.load(DATA/'train_x.npz'))
data=np.load(OUT/'datos_semi_markov/runs_train.npz')
g=read(OUT/'datos_semi_markov/protocolo.json')
N=g['N']
cell=data['cell'].astype(np.int32)
duration=data['duration_n']
enter=data['enter_n']
offset=data['run_offsets']
trip=data['trip_train_index']
valid=np.unpackbits(data['outgoing_valid_bits'])[:len(cell)].astype(bool)
xx=(tr['d_lon']+8.62)*111195*np.cos(np.deg2rad(41.16))
yy=(tr['d_lat']-41.16)*111195
dest=np.floor((xx-g['x0'])/200).astype(np.int32)*g['ny']+np.floor((yy-g['y0'])/200).astype(np.int32)
base_groups=N*5*2
leaf_groups=base_groups*12
fit_hist=np.zeros((leaf_groups,H,2),np.uint32)
held_hist=np.zeros_like(fit_hist)
held_group=[];held_age=[];held_event=[];held_day=[]
kept=0;fit_runs=0;held_runs=0
for first in range(0,len(cell),500000):
    rr=np.arange(first,min(first+500000,len(cell)))
    ti=np.searchsorted(offset,rr,side='right')-1
    ii=trip[ti]
    cur=cell[rr]
    previous=cell[np.maximum(rr-1,0)]
    terminal=rr==offset[ti+1]-1
    known=(rr>offset[ti])&(previous<N)&valid[np.maximum(rr-1,0)]
    dx=cur//g['ny']-previous//g['ny']
    dy=cur%g['ny']-previous%g['ny']
    heading=np.where(known,np.where(np.abs(dx)>=np.abs(dy),np.where(dx>=0,0,1),np.where(dy>=0,2,3)),4)
    at_destination=cur==dest[ii]
    age=duration[rr].astype(np.int16)-(~terminal)
    good=(cur<N)&(age>=0)&(age<H)
    assert np.all((~terminal)|at_destination)
    timestamp=tr['timestamp'][ii]
    entered=timestamp+enter[rr].astype(np.int64)*15
    weekday=(entered//86400+3)%7
    clock=(entered%86400)//14400+6*(weekday>=5)
    base=(cur*5+heading)*2+at_destination
    group=base*12+clock
    hold=(timestamp//(7*86400))%5==0
    event=terminal.astype(np.uint8)
    a=good&(~hold);b=good&hold
    np.add.at(fit_hist,(group[a],age[a],event[a]),1)
    np.add.at(held_hist,(group[b],age[b],event[b]),1)
    held_group.append(group[b].astype(np.int32))
    held_age.append(age[b].astype(np.uint8))
    held_event.append(event[b])
    held_day.append((timestamp[b]//86400).astype(np.int32))
    kept+=int(good.sum());fit_runs+=int(a.sum());held_runs+=int(b.sum())
    if first%2000000==0:print(json.dumps(dict(stage='Train_only_hazard_counts',completed_runs=first,total_runs=len(cell))),flush=True)
hg=np.concatenate(held_group)
ha=np.concatenate(held_age).astype(np.int64)
he=np.concatenate(held_event).astype(np.int64)
hd=np.concatenate(held_day)
del held_group,held_age,held_event,held_day,data,cell,duration,enter,valid,tr
base_fit=fit_hist.reshape(base_groups,12,H,2).sum(1,dtype=np.uint32)
global_fit=base_fit.reshape(N,5,2,H,2).sum((0,1),dtype=np.uint32)
global_counts=point_counts(global_fit)
prior=np.array([[.5,.5,0.],[.49,.49,.02]])
pglobal=(global_counts+prior[:,None,:])/(global_counts.sum(-1,keepdims=True)+1)
base_counts=point_counts(base_fit)
base_flag=np.arange(base_groups)%2
pbase=((base_counts+50*pglobal[base_flag])/(base_counts.sum(-1,keepdims=True)+50)).astype(np.float32)
scores={}
losses={}
losses['global_age']=run_losses(pglobal,(hg//12)%2,ha,he)
losses['cell_heading_destination_age']=run_losses(pbase,hg//12,ha,he)
pclock=np.empty((leaf_groups,H,3),np.float32)
noage=np.empty((leaf_groups,3),np.float32)
# Fixed smoothing strengths, no tuning on held fold.
global_noage=(global_counts.sum(1)+prior)/(global_counts.sum((1,2))[:,None]+1)
base_noage=(base_counts.sum(1)+50*global_noage[base_flag])/(base_counts.sum((1,2))[:,None]+50)
for first in range(0,leaf_groups,4096):
    last=min(first+4096,leaf_groups)
    c=point_counts(fit_hist[first:last])
    par=pbase[np.arange(first,last)//12]
    pclock[first:last]=((c+20*par)/(c.sum(-1,keepdims=True)+20)).astype(np.float32)
    cn=c.sum(1)
    pp=base_noage[np.arange(first,last)//12]
    noage[first:last]=(cn+20*pp)/(cn.sum(-1,keepdims=True)+20)
    if first%65536==0:print(json.dumps(dict(stage='hazard_hierarchy',completed_groups=first,total_groups=leaf_groups)),flush=True)
losses['cell_heading_destination_clock_age']=run_losses(pclock,hg,ha,he)
# A constant hazard at every age, with exactly the same context hierarchy.
nlp=np.log(np.maximum(noage,1e-30)).astype(np.float64)
losses['cell_heading_destination_clock_no_age']=-ha*nlp[hg,0]-nlp[hg,he+1]
points=ha+1
denominator=int(points.sum())
for name,loss in losses.items():scores[name]=dict(held_NLL_per_point=float(loss.sum()/denominator),held_runs=len(loss),held_points=denominator)
def compare(a,b):
    diff=losses[a]-losses[b]
    days,di=np.unique(hd,return_inverse=True)
    sums=np.bincount(di,weights=diff)
    counts=np.bincount(di,weights=points)
    boot=np.random.default_rng(1729).integers(len(days),size=(3000,len(days)))
    ci=np.quantile(sums[boot].sum(1)/counts[boot].sum(1),[.025,.975])
    return dict(delta_NLL_per_point=float(diff.sum()/denominator),CI95_by_Train_day=ci,days=len(days))
clock_effect=compare('cell_heading_destination_clock_age','cell_heading_destination_age')
age_effect=compare('cell_heading_destination_clock_age','cell_heading_destination_clock_no_age')
# Verify aggregate exposure accounting independently of individual-run loss evaluation.
held_exposures=0
for first in range(0,leaf_groups,4096):held_exposures+=int(point_counts(held_hist[first:first+4096]).sum())
assert held_exposures==denominator
assert np.allclose(pclock.sum(-1),1,atol=2e-7)
assert not torch.cuda.is_initialized()
report=dict(passed=True,protocol=dict(source='Train GPS only; all observed states, no Validation or Test GPS or labels',held='Train departure UTCweek modulo5==0; every count and backoff excludes this complete fold',features='Cell, incoming heading from past, known destination equality, run-entry UTC4h bin and weekend; run entry computed from departure plus past elapsed time',objective='Proper three-event stay/move/end likelihood; no observed query length as predictor',smoothing='Global Dirichlet1; cell/heading/destination50; clock20 fixed before held evaluation',clock='UTC; distinguish from Lisbon local clock; no claim of exact UNITE reproduction',age='Elapsed dwell age from past; at inference age and elapsed time would be model-generated',scope='Local likelihood instrument, not an achieved OD duration MAE; training cohort and labels are unchanged'),fit_runs=fit_runs,held_runs=held_runs,kept_runs=kept,scores=scores,clock_effect=clock_effect,age_effect=age_effect,worth_temporal_hazard_prior_test=bool(clock_effect['delta_NLL_per_point']<-.001 and clock_effect['CI95_by_Train_day'][1]<0),max_fit_event_bin=int(fit_hist.max()),max_held_event_bin=int(held_hist.max()),no_GPU=True,no_histogram_bank_saved=True,only_Train_files_opened=True,Test_access=False,actual_files_accessed=sorted(access),code_sha256=sha(__file__),seconds=time.time()-start)
write(OUT/'diagnostico_riesgos_temporales.json',report)
print(json.dumps({k:report[k] for k in ['scores','clock_effect','age_effect','worth_temporal_hazard_prior_test','seconds']},default=lambda x:x.tolist()),flush=True)
