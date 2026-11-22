"""CPU pilot: shared trip environment versus independent marginal variation. Train only."""
from base_replanteo import *
from modelo_dir_fases import PhaseDirectionalArrival
from modelo_direccional import heading_ids
from verosimilitud_entorno import environment_log_likelihood
from inferencia import feature_input
import torch,math

torch.set_num_threads(4)
torch.manual_seed(2026)
access=deny_test()
parent=OUT/'espacial_fases_referencia_s0'
assert (parent/'resultado.json').exists(),'Wait for the physical model to be frozen'
run=OUT/'piloto_entorno_compartido_s0'
assert not run.exists()
run.mkdir()
start=time.time()
source_sha=sha(parent/'modelo.pt')
ck=torch.load(parent/'modelo.pt',map_location='cpu',weights_only=False)
grid=ck['grid']
graph=dict(np.load(OUT/'datos_semi_markov/grafo_grid.npz'))
distance=np.load(OUT/'datos_semi_markov/distancias_grid.npy')
counts=np.load(OUT/'prior_transiciones/conteos_fold.npz')['counts']
model=PhaseDirectionalArrival(ck['stats'],grid,graph,distance,ck['power'],counts,ck['phase_initialization']).eval()
model.load_state_dict(ck['state'])
lags=torch.tensor(ck['phase_initialization']['lags'])
for name in ['counts_fold','counts_base_fold','counts_global_fold']:setattr(model,name,torch.empty(0))
del ck,counts
tr,X,names=feature_input('train',read(BEST/'resultado.json'))
data=np.load(OUT/'datos_semi_markov/runs_train.npz')
cells=data['cell'].astype(np.int32)
duration=data['duration_n'].astype(np.int64)
offset=data['run_offsets']
trip=data['trip_train_index']
valid=np.unpackbits(data['outgoing_valid_bits'])[:len(cells)].astype(bool)
rng=np.random.default_rng(2026)
selected_trips=[]
selected_runs=[]
for ti in rng.permutation(len(trip)):
    rr=np.arange(offset[ti],offset[ti+1]-1)
    rr=rr[valid[rr]&(cells[rr]<model.N)&(cells[rr+1]<model.N)]
    if len(rr)<2:continue
    rr=np.sort(rng.choice(rr,min(8,len(rr)),replace=False))
    selected_trips.append(ti)
    selected_runs.append(rr)
    if len(selected_trips)==25000:break
assert len(selected_trips)==25000
indices=trip[np.array(selected_trips)]
R=8
B=len(indices)
group=np.concatenate([np.full(len(rr),i,np.int64) for i,rr in enumerate(selected_runs)])
position=np.concatenate([np.arange(len(rr)) for rr in selected_runs])
rows=np.concatenate(selected_runs)
wcache=torch.zeros(B,R,8)
hcache=torch.zeros_like(wcache)
mask=torch.zeros(B,R,dtype=torch.bool)
dur=torch.ones(B,R,dtype=torch.long)
def ids(tag):
    xx=(tr[tag+'_lon']+8.62)*111195*np.cos(np.deg2rad(41.16))
    yy=(tr[tag+'_lat']-41.16)*111195
    return np.floor((xx-grid['x0'])/200).astype(np.int64)*grid['ny']+np.floor((yy-grid['y0'])/200).astype(np.int64)
origin=ids('o')
destination=ids('d')
queries=[]
with torch.no_grad():
    for first in range(0,B,512):
        ii=indices[first:first+512]
        q,*_=model.encode_od(torch.tensor(X[ii]),torch.tensor(origin[ii]),torch.tensor(destination[ii]))
        queries.append(q)
    qcache=torch.cat(queries)
    for first in range(0,len(rows),1024):
        rr=rows[first:first+1024]
        gg=group[first:first+1024]
        pp=position[first:first+1024]
        ii=indices[gg]
        ti=np.array(selected_trips)[gg]
        cur=torch.tensor(cells[rr],dtype=torch.long)
        prev=torch.tensor(cells[np.maximum(rr-1,0)],dtype=torch.long)
        known=torch.tensor((rr>offset[ti])&(cells[np.maximum(rr-1,0)]<model.N)&valid[np.maximum(rr-1,0)])
        heading=torch.where(known,heading_ids(cur//grid['ny']-prev//grid['ny'],cur%grid['ny']-prev%grid['ny']),4)
        _,w,h=model.phase_parameters(qcache[gg],torch.tensor(X[ii,:4]),cur,torch.tensor(destination[ii]),heading)
        wcache[gg,pp]=w
        hcache[gg,pp]=h
        mask[gg,pp]=True
        dur[gg,pp]=torch.tensor(duration[rr])
        if first%32768==0:print(json.dumps(dict(stage='Train_only_residence_encoding',completed=first,total=len(rows))),flush=True)
del model,X,graph,distance,qcache
hcache=hcache.clamp(.001,.999)
logw=wcache.clamp_min(1e-30).log()
logit=torch.logit(hcache)
days=tr['timestamp'][indices]//86400
fold=(tr['timestamp'][indices]//(7*86400))%5
fit=torch.tensor(np.flatnonzero(fold!=0))
held=torch.tensor(np.flatnonzero(fold==0))
nodes,weights=np.polynomial.hermite.hermgauss(4)
nodes=torch.tensor(nodes,dtype=torch.float32)
weights=torch.tensor(weights/math.sqrt(math.pi),dtype=torch.float32)
protocol=dict(parent_sha256=source_sha,sample_trips=B,observed_runs=len(rows),up_to8_distinct_runs_per_trip=True,fit_trips=len(fit),held_trips=len(held),held_rule='Train UTCweek modulo5==0',controls=['location-only sigma0','independent environment each residence','one shared environment per trip'],only_scalar_controls_are_heldout='Frozen representation itself saw eligible Train; not independent whole-model validation',Validation_or_Test_GPS_or_labels_used=False,Test_access=False)
write(run/'protocolo.json',protocol)
def values(kind,ii,location,sigma):
    mode='shared' if kind=='shared' else 'independent'
    return environment_log_likelihood(logw[ii],logit[ii],dur[ii],mask[ii],lags,location,sigma,nodes,weights,mode)
results={}
held_values={}
for kind in ['baseline','independent','shared']:
    bias=torch.nn.Parameter(torch.tensor(0.))
    raw_sigma=torch.nn.Parameter(torch.tensor(math.log(math.expm1(.1))))
    parameters=[bias] if kind=='baseline' else [bias,raw_sigma]
    optimizer=torch.optim.Adam(parameters,lr=.03)
    best=float('inf')
    best_parameters=None
    bad=0
    history=[]
    for step in range(1,241):
        optimizer.zero_grad()
        location=2*bias.tanh()
        sigma=torch.tensor(0.) if kind=='baseline' else torch.nn.functional.softplus(raw_sigma).clamp_max(1.5)
        ll=values(kind,fit,location,sigma)
        loss=-ll.sum()/mask[fit].sum()
        assert torch.isfinite(loss)
        loss.backward()
        candidate_parameters=(bias.detach().clone(),raw_sigma.detach().clone())
        optimizer.step()
        score=float(loss.detach())
        if score<best-1e-7:
            best=score
            best_parameters=candidate_parameters
            bad=0
        else:bad+=1
        if step%40==0:
            history.append(dict(step=step,fit_NLL_nats=score))
            print(json.dumps(dict(control=kind,step=step,fit_NLL_nats=score)),flush=True)
        if bad>=40:break
    with torch.no_grad():
        bias,raw_sigma=best_parameters
        location=2*bias.tanh()
        sigma=torch.tensor(0.) if kind=='baseline' else torch.nn.functional.softplus(raw_sigma).clamp_max(1.5)
        hv=values(kind,held,location,sigma)
        held_values[kind]=hv.numpy()
        results[kind]=dict(fit_NLL_per_observed_run=best,held_NLL_per_observed_run=float(-hv.sum()/mask[held].sum()),location=float(location),sigma=float(sigma),steps=step,history=history)
    write(run/'resultados_parciales.json',results)
diff=-held_values['shared']+held_values['independent']
held_days=days[held.numpy()]
_,di=np.unique(held_days,return_inverse=True)
s=np.bincount(di,weights=diff)
n=np.bincount(di,weights=mask[held].sum(-1).numpy())
boot=np.random.default_rng(1729).integers(len(n),size=(3000,len(n)))
ci=np.quantile(s[boot].sum(1)/n[boot].sum(1),[.025,.975])
delta=float(s.sum()/n.sum())
assert not torch.cuda.is_initialized()
assert sha(parent/'modelo.pt')==source_sha
report=dict(protocol=protocol,results=results,shared_minus_independent_held_NLL_nats=delta,CI95_by_Train_day=ci,worth_GPU_shared_effect_test=bool(delta<-.002 and ci[1]<0),interpretation='Only a likelihood test conditional on actual Train teaching states. More flexibility alone is controlled by independent variation. Even a positive result does not establish OD travel-time MAE improvement.',no_GPU_used=True,no_representation_bank_on_disk=True,sample_runs_sha256=hashlib.sha256(rows.tobytes()).hexdigest(),code_sha256=sha(__file__),likelihood_sha256=sha(OUT/'verosimilitud_entorno.py'),Test_access=False,seconds=time.time()-start)
write(run/'resultado.json',report)
print(json.dumps({k:report[k] for k in ['results','shared_minus_independent_held_NLL_nats','CI95_by_Train_day','worth_GPU_shared_effect_test','seconds']},default=lambda x:x.tolist()),flush=True)

